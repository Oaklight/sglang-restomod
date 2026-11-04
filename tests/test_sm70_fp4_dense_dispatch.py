"""CPU test for the dense NVFP4 bridge into ``marlin_utils_fp4``.

Every name this component replaces is looked up in that module's own namespace,
so the test swaps a fake module in and drives the patches directly: nothing here
needs a GPU or a built extension. What is pinned is the wiring -- that all four
names are replaced, that a second install is a no-op, and the argument order of
the two bridges, where a shifted slot produces numbers that look plausible and
are wrong.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp4_dense import method  # noqa: E402

_BRIDGES = (
    "gptq_marlin_gemm",
    "gptq_marlin_repack",
    "marlin_permute_scales",
    "nvfp4_marlin_process_global_scale",
)


class _ScalarType:
    """Stands in for ``scalar_types.float4_e2m1f``: the bridge only reads ``id``."""

    id = 1049600


@pytest.fixture
def fp4(monkeypatch):
    calls: dict[str, object] = {}

    def gemm(*args):
        calls["gemm"] = args
        return "gemm-result"

    def repack(*args):
        calls["repack"] = args
        return "repack-result"

    def permute(s, size_k, size_n, group_size):
        calls["permute"] = (size_k, size_n, group_size)
        return "permuted-result"

    def global_scale(value):
        calls["global_scale"] = value
        return value

    def prepare(layer):
        calls["prepare"] = layer

    module = types.ModuleType("sglang.srt.layers.quantization.marlin_utils_fp4")
    module.gptq_marlin_gemm = gemm
    module.gptq_marlin_repack = repack
    module.marlin_permute_scales = permute
    module.nvfp4_marlin_process_global_scale = global_scale
    module.prepare_nvfp4_layer_for_marlin = prepare

    quantization = types.ModuleType("sglang.srt.layers.quantization")
    quantization.marlin_utils_fp4 = module
    layers = types.ModuleType("sglang.srt.layers")
    layers.quantization = quantization
    srt = types.ModuleType("sglang.srt")
    srt.layers = layers
    sglang = types.ModuleType("sglang")
    sglang.srt = srt

    for name, mod in (
        ("sglang", sglang),
        ("sglang.srt", srt),
        ("sglang.srt.layers", layers),
        ("sglang.srt.layers.quantization", quantization),
        ("sglang.srt.layers.quantization.marlin_utils_fp4", module),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(method, "available", lambda: True)
    monkeypatch.setattr(method, "ops", lambda: (repack, gemm))
    monkeypatch.setattr(method, "_COLLAPSE_WARNED", False)

    return types.SimpleNamespace(
        module=module, calls=calls, gemm=gemm, repack=repack, permute=permute
    )


def test_install_replaces_every_name_the_path_resolves(fp4):
    assert method.install() is True

    assert fp4.module.gptq_marlin_gemm is not fp4.gemm
    assert fp4.module.gptq_marlin_repack is not fp4.repack
    assert fp4.module.marlin_permute_scales is method.marlin_permute_scales
    for name in _BRIDGES:
        assert getattr(fp4.module, name)._restomod_sm70 is True


def test_a_second_install_changes_nothing(fp4):
    assert method.install() is True
    installed = {name: getattr(fp4.module, name) for name in _BRIDGES}

    assert method.install() is False
    assert {name: getattr(fp4.module, name) for name in _BRIDGES} == installed


def test_repack_carries_the_extension_arguments_in_order(fp4):
    method.install()
    weight, perm = object(), object()

    assert fp4.module.gptq_marlin_repack(weight, perm, 4096, 5120) == "repack-result"
    assert fp4.calls["repack"] == (weight, perm, 4096, 5120, 4, False)


def test_repack_refuses_a_width_the_kernel_cannot_read(fp4):
    method.install()

    with pytest.raises(ValueError, match="packs 4-bit tiles"):
        fp4.module.gptq_marlin_repack(object(), object(), 4096, 5120, num_bits=8)


def test_scales_are_left_unpermuted(fp4):
    method.install()
    scales = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    out = fp4.module.marlin_permute_scales(scales, 64, 3, 16)

    assert torch.equal(out, scales.reshape(-1, 3))
    assert out.is_contiguous()
    assert "permute" not in fp4.calls


def test_gemm_maps_every_slot(fp4):
    method.install()
    a, c, w, b_scales, b_zeros, g_idx, perm, workspace = (object() for _ in range(8))
    global_scale = object()

    assert (
        fp4.module.gptq_marlin_gemm(
            a,
            c,
            w,
            b_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            workspace,
            _ScalarType(),
            6,
            5120,
            4096,
            True,
            False,
            True,
            False,
        )
        == "gemm-result"
    )

    args = fp4.calls["gemm"]
    assert args[:3] == (a, c, w)
    assert args[3] is None  # bias: an extension slot the engine does not carry
    assert args[4] is b_scales
    assert args[5] is None  # activation scales: the same
    assert args[6] is global_scale
    assert args[7:11] == (b_zeros, g_idx, perm, workspace)
    assert args[11] == _ScalarType.id  # b_q_type.id, which selects NVFP4
    assert args[12:16] == (6, 5120, 4096, True)
    assert args[16:19] == (False, True, False)


def test_zero_rows_returns_an_empty_output_without_the_kernel(fp4):
    method.install()
    a = torch.zeros((0, 4096), dtype=torch.float16)

    out = fp4.module.gptq_marlin_gemm(
        a,
        None,
        object(),
        object(),
        None,
        None,
        None,
        None,
        object(),
        _ScalarType(),
        0,
        5120,
        4096,
        True,
        False,
        False,
        False,
    )

    assert out.shape == (0, 5120)
    assert out.dtype == torch.float16
    assert "gemm" not in fp4.calls


def test_a_fused_global_scale_is_collapsed_to_one_half(fp4):
    method.install()
    per_shard = torch.tensor([1.5, 2.5], dtype=torch.float32)

    fp4.module.nvfp4_marlin_process_global_scale(per_shard)

    passed = fp4.calls["global_scale"]
    assert passed.numel() == 1
    assert passed.dtype == torch.float16
    assert passed.item() == pytest.approx(2.5)


def test_a_scalar_global_scale_is_only_cast(fp4):
    method.install()

    fp4.module.nvfp4_marlin_process_global_scale(torch.tensor([0.25], dtype=torch.float32))

    passed = fp4.calls["global_scale"]
    assert passed.numel() == 1
    assert passed.dtype == torch.float16
    assert passed.item() == pytest.approx(0.25)


def test_a_missing_extension_refuses_at_weight_load(fp4, monkeypatch):
    monkeypatch.setattr(method, "available", lambda: False)

    assert method.install() is False
    assert fp4.module.prepare_nvfp4_layer_for_marlin._restomod_sm70 is True
    with pytest.raises(RuntimeError, match="needs a loadable marlin_v100 build"):
        fp4.module.prepare_nvfp4_layer_for_marlin(object())
