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
    monkeypatch.delenv(decode_qpn2.ENV_ENABLE, raising=False)
    assert decode_qpn2.supports(4, 320, 2560, 10, 8)
    assert decode_qpn2.supports(8, 2560, 160, 10, 8)
    assert not decode_qpn2.supports(64, 320, 2560, 10, 8), "prefill is not covered"
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
    # The kernel does not cover M > 8, and Marlin cannot read a QPN2 buffer.
    with pytest.raises(RuntimeError, match="prefill reader is not integrated"):
        _qpn2_call(size_m=64, block=8)


def test_qpn2_weight_with_a_non_decode_block_raises():
    with pytest.raises(RuntimeError, match="prefill reader is not integrated"):
        _qpn2_call(size_m=4, block=16)
