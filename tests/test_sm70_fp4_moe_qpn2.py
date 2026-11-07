"""CPU tests for the QPN2 layout switch, repack, and prefill refusal (#224).

Runs in `restomod-ci` with CPU torch only: no GPU, no sglang engine, no
marlin_v100 extension. Covers the parts that decide what the server does
before any kernel runs:

* the load-time switch (`SGLANG_SM70_FP4_MOE_LAYOUT`, `..._QPN2_TENSORS`);
* the call-time layout marker (uint8 QPN2 vs int32 Marlin);
* the repack against the CPU-verified reference in `scripts/qpn2_probe`;
* the decode shape gate, and the raise that keeps a QPN2 buffer away from a
  reader that cannot decode it (prefill, or a non-decode route block).
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp4_moe import decode_qpn2, qpn2  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "scripts" / "qpn2_probe" / "repack_qpn2.py"
LAYOUT_ENV = "SGLANG_SM70_FP4_MOE_LAYOUT"
TENSORS_ENV = "SGLANG_SM70_QPN2_TENSORS"


def _reference_module():
    spec = importlib.util.spec_from_file_location("qpn2_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _e4m3_bytes(n: int, groups: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    raw = torch.randint(0, 8, (n, groups), generator=generator, dtype=torch.int32)
    raw |= torch.randint(2, 11, (n, groups), generator=generator, dtype=torch.int32) << 3
    return raw.to(torch.uint8)


@pytest.mark.parametrize("size_n,size_k", [(32, 16), (64, 32), (320, 64), (64, 160)])
def test_pack_expert_matches_reference(size_n, size_k):
    reference = _reference_module()
    generator = torch.Generator().manual_seed(3)
    packed = torch.randint(
        0, 256, (size_n, size_k // 2), generator=generator, dtype=torch.int32
    ).to(torch.uint8)
    scales = _e4m3_bytes(size_n, size_k // 16).view(torch.float8_e4m3fn)

    want = reference.pack_qpn2(packed, scales, size_n, size_k)
    got = qpn2.pack_expert(packed, scales, size_n, size_k)

    assert got.numel() == (size_n // 32) * (size_k // 16) * qpn2.GROUP_BYTES
    assert torch.equal(got, want)


def test_layout_switch_defaults_to_marlin(monkeypatch):
    monkeypatch.delenv(LAYOUT_ENV, raising=False)
    monkeypatch.delenv(TENSORS_ENV, raising=False)
    assert qpn2.layout() == "marlin"
    assert qpn2.selected_prefixes() == set()


def test_layout_switch_selects_tensors(monkeypatch):
    monkeypatch.setenv(LAYOUT_ENV, "qpn2")
    monkeypatch.setenv(TENSORS_ENV, "both")
    assert qpn2.selected_prefixes() == {"w13", "w2"}
    monkeypatch.setenv(TENSORS_ENV, "w13")
    assert qpn2.selected_prefixes() == {"w13"}
    monkeypatch.setenv(TENSORS_ENV, "w2")
    assert qpn2.selected_prefixes() == {"w2"}


@pytest.mark.parametrize("value", ["fp8", "qpn4", "w1"])
def test_layout_switch_rejects_unknown_values(monkeypatch, value):
    monkeypatch.setenv(LAYOUT_ENV, "qpn2")
    monkeypatch.setenv(TENSORS_ENV, value)
    with pytest.raises(ValueError):
        qpn2.selected_prefixes()
    monkeypatch.setenv(LAYOUT_ENV, value)
    with pytest.raises(ValueError):
        qpn2.layout()


def test_layout_marker_is_the_dtype():
    assert qpn2.is_qpn2_weight(torch.zeros(8, 460800, dtype=torch.uint8))
    assert not qpn2.is_qpn2_weight(torch.zeros(8, 160, 640, dtype=torch.int32))
    assert not qpn2.is_qpn2_weight(torch.zeros(460800, dtype=torch.uint8))


def test_supports_covers_the_decode_regime(monkeypatch):
    # The argument is the routed-row count the kernel runs on, not size_m:
    # upstream passes tokens for the first GEMM and M*topk for the second, and
    # gating on size_m rejected the down projection of an ordinary decode batch
    # (rows=40, top_k=1) as if it were a 40-token prefill.
    monkeypatch.delenv(decode_qpn2.ENV_ENABLE, raising=False)
    assert decode_qpn2.supports(4, 320, 2560, 10, 8), "gate_up, short batch"
    assert decode_qpn2.supports(40, 2560, 160, 1, 8), "down projection of 5x8 routes"
    assert not decode_qpn2.supports(0, 320, 2560, 10, 8), "no routes"
    assert not decode_qpn2.supports(
        decode_qpn2._WORKSPACE_ROUTE_TILES + 1, 320, 2560, 10, 8
    ), "above the workspace bound"
    assert not decode_qpn2.supports(4, 320, 2560, 10, 64), "prefill route block"
    assert not decode_qpn2.supports(4, 320, 2560, 10, 16), "route block is 8"
    assert not decode_qpn2.supports(4, 320, 2552, 10, 8), "K must be a multiple of 16"
    assert not decode_qpn2.supports(4, 312, 2560, 10, 8), "N must be a multiple of 32"
    monkeypatch.setenv(decode_qpn2.ENV_ENABLE, "0")
    assert not decode_qpn2.supports(4, 320, 2560, 10, 8)


def test_split_k_follows_the_group_count():
    assert decode_qpn2.split_k_for(2560) == decode_qpn2.SPLIT_K_LONG
    assert decode_qpn2.split_k_for(160) == decode_qpn2.SPLIT_K_SHORT
    assert decode_qpn2.split_k_for(1024) == decode_qpn2.SPLIT_K_LONG


def _qpn2_call(size_m: int, block: int):
    """A shipping-entry call whose weight buffer is QPN2 (CPU tensors)."""
    from sglang_restomod.quant.sm70_fp4_moe import method

    size_n, size_k, experts = 320, 2560, 2
    rows = size_m * 2
    tiles, groups = size_n // 32, size_k // 16
    weights = torch.zeros(
        (experts, tiles * groups * qpn2.GROUP_BYTES), dtype=torch.uint8
    )
    return method.moe_wna16_marlin_gemm(
        torch.zeros(size_m, size_k, dtype=torch.float16),
        torch.zeros(rows, size_n, dtype=torch.float16),
        weights,
        None,
        torch.zeros(experts, size_n, groups, dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.ones(experts, dtype=torch.float32),
        None,
        None,
        None,
        torch.empty(0, dtype=torch.int32),
        torch.zeros(rows * block, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.tensor([rows * block], dtype=torch.int32),
        torch.ones(rows, dtype=torch.float32),
        block,
        2,
        False,
        False,
        None,
        size_m,
        size_n,
        size_k,
    )


def test_qpn2_weight_on_a_prefill_shape_raises():
    # The decode kernel does not cover M > 8. The prefill fallback repacks into
    # the Marlin scratch, so the call raises only when that fallback is not
    # available (as here, with no extension); either way the QPN2 buffer never
    # reaches a reader that cannot decode it.
    with pytest.raises(RuntimeError, match="QPN2 expert weights reached a shape"):
        _qpn2_call(size_m=64, block=8)


def test_qpn2_weight_with_a_non_decode_block_raises():
    with pytest.raises(RuntimeError, match="QPN2 expert weights reached a shape"):
        _qpn2_call(size_m=4, block=16)


def _stub_copy_or_rebind(monkeypatch):
    """CPU stand-in for the engine helper, so the test needs no sglang.

    The engine's copy_or_rebind_param rebinds when the shape or dtype changes
    (pin-site sglang/srt/layers/utils/common.py); this is that branch.
    """
    import sys
    import types

    module = types.ModuleType("sglang.srt.layers.utils")

    def copy_or_rebind_param(target, name, new_value):
        new_value = new_value.detach()
        param = getattr(target, name, None)
        if isinstance(param, torch.nn.Parameter):
            if param.data.shape == new_value.shape and param.data.dtype == new_value.dtype:
                param.data.copy_(new_value)
            else:
                param.data = new_value
            param.requires_grad_(False)
        else:
            setattr(target, name, torch.nn.Parameter(new_value, requires_grad=False))

    module.copy_or_rebind_param = copy_or_rebind_param
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils", module)


def _layer_pointers(layer):
    """Every buffer the layer still holds, by storage address."""
    return {
        value.data_ptr()
        for value in vars(layer).values()
        if isinstance(value, torch.Tensor)
    }


def test_prepare_prefix_drops_the_duplicate_scale_tensor(monkeypatch):
    """QPN2 keeps the Marlin footprint: the bank carries the group scales.

    The 288-byte group stores the 32 E4M3 scale bytes inline, so the
    checkpoint's ``[E, N, K/16]`` scale tensor is a duplicate and must be
    released at install time. Pinned here because restoring it would still pass
    every numeric test: the call sites never read the attribute on the QPN2
    path, so only a footprint assertion can catch the regression.
    """
    from types import SimpleNamespace

    from sglang_restomod.quant.sm70_fp4_moe import qpn2_repack

    _stub_copy_or_rebind(monkeypatch)
    monkeypatch.setenv(LAYOUT_ENV, "qpn2")
    monkeypatch.setenv(TENSORS_ENV, "both")
    qpn2_repack.reset()

    experts, size_n, size_k = 2, 64, 32
    generator = torch.Generator().manual_seed(5)
    packed = torch.randint(
        0, 256, (experts, size_n, size_k // 2), generator=generator, dtype=torch.int32
    ).to(torch.uint8)
    scale_bytes = _e4m3_bytes(size_n, size_k // 16).repeat(experts, 1, 1)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(packed, requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(
            scale_bytes.view(torch.float8_e4m3fn), requires_grad=False
        ),
        w13_weight_scale_2=torch.nn.Parameter(
            torch.full((experts,), 0.01), requires_grad=False
        ),
    )

    source_pointers = {layer.w13_weight.data_ptr(), layer.w13_weight_scale.data_ptr()}
    qpn2.prepare_prefix(layer, "w13")

    tiles, groups = size_n // 32, size_k // 16
    assert layer.w13_weight.dtype is torch.uint8
    assert tuple(layer.w13_weight.shape) == (
        experts,
        tiles * groups * qpn2.GROUP_BYTES,
    )
    # The duplicate is gone, not merely dereferenced: the installed parameter is
    # empty and nothing on the layer still points at the checkpoint buffers.
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w13_weight_scale.dtype is torch.float8_e4m3fn
    assert not (source_pointers & _layer_pointers(layer))
    qpn2_repack.reset()


def test_qpn2_shape_patch_is_present_and_derives_correctly():
    """Pin the upstream patch that lets a QPN2 bank reach the wrapper.

    A QPN2 server died during CUDA graph capture because ``fused_marlin_moe``
    asserts the Marlin weight layout before it calls the one function this
    plugin replaces. The fix is patch 0003, which derives E/K/N from the bank
    when ``w1`` is uint8. This test keeps the patch in the tree (dropping it
    would silently restore the capture failure) and checks that the derivation
    it uses reproduces N from the bank's size for the shipped shapes.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    patch = (root / "patches" / "0003-qpn2-marlin-moe-shape.patch").read_text()
    assert "w1.dtype == torch.uint8" in patch

    manifest = json.loads((root / "patches" / "manifest.json").read_text())
    entry = manifest["patches"]["0003-qpn2-marlin-moe-shape.patch"]
    assert entry["paths"] == [
        "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"
    ]

    for size_n, size_k in ((320, 2560), (2560, 160), (64, 32)):
        experts = 2
        per_expert = (size_n // 32) * (size_k // 16) * qpn2.GROUP_BYTES
        # The patch computes N from the bank's per-expert size and K.
        assert (experts * per_expert // experts) * 16 // (9 * size_k) == size_n


def test_decode_fallback_warning_predicate():
    """The warning must fire for a decode step, not for a short prefill chunk.

    A prefill chunk small enough for the align heuristic to pick block 8 is
    still a prefill; its size_m is the chunk's route count. Only a decode
    step's size_m is one batch's routes (M * topk, tens of rows).
    """
    from sglang_restomod.quant.sm70_fp4_moe.method import _is_decode_fallback

    assert _is_decode_fallback(10, 8), "1 token x topk 10 at decode"
    assert _is_decode_fallback(64, 8), "at the bound"
    assert not _is_decode_fallback(1280, 8), "a short prefill chunk, block 8"
    assert not _is_decode_fallback(40960, 8), "a long prefill chunk"
    assert not _is_decode_fallback(10, 64), "prefill route block"
