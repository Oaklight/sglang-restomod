"""Shape guards for the SM70 block-FP8 dense and MoE paths (CPU).

The packed E4M3 kernel spans complete K quantization groups and writes N in
32-column HMMA-884 tiles. A partial group does not raise in the kernel: it
corrupts the packing or wedges split-K completion. These cases pin the Python
side that refuses the shape first, so the GPU suite only has to cover what the
kernel accepts.

Everything here runs before the CUDA operators are touched, so the module is
part of the default CPU CI selection.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_fp8_dense import _method as kernels  # noqa: E402
from sglang_restomod.quant.sm70_fp8_moe import (  # noqa: E402
    Experts,
    _check_call_matches_prepared,
    _check_gemm_shape,
    gemm,
    prepare_experts,
)


def _layer(n=256, k=256, dtype=torch.float16):
    layer = torch.nn.Module()
    layer.orig_dtype = dtype
    layer.weight_block_size = [128, 128]
    layer.weight = torch.nn.Parameter(
        torch.zeros(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(n // 128, k // 128), requires_grad=False
    )
    layer.output_size_per_partition = n
    return layer


# --- dense weight shape -----------------------------------------------------


def test_aligned_dense_shape_is_accepted():
    kernels._check_dense_weight_shape(256, 256)


def test_dense_n_that_is_not_a_32_multiple_is_refused():
    with pytest.raises(RuntimeError, match="N a multiple of 32"):
        kernels._check_dense_weight_shape(104, 256)


def test_dense_partial_k_group_is_refused():
    with pytest.raises(RuntimeError, match="quantization group"):
        kernels._check_dense_weight_shape(256, 192)


def test_dense_empty_shape_is_refused():
    with pytest.raises(RuntimeError, match="positive N and K"):
        kernels._check_dense_weight_shape(0, 256)
    with pytest.raises(RuntimeError, match="positive N and K"):
        kernels._check_dense_weight_shape(256, 0)


def test_prepare_refuses_an_unaligned_weight_before_packing():
    layer = _layer(n=104, k=256)
    with pytest.raises(RuntimeError, match="N a multiple of 32"):
        kernels.prepare_sm70_turbomind_fp8_linear(layer)


def test_prepare_refuses_a_partial_k_group_before_packing():
    layer = _layer(n=256, k=192)
    with pytest.raises(RuntimeError, match="quantization group"):
        kernels.prepare_sm70_turbomind_fp8_linear(layer)


def test_prepare_refuses_scales_that_do_not_cover_the_weight():
    layer = _layer(n=256, k=256)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
    with pytest.raises(RuntimeError, match="requires scales of"):
        kernels.prepare_sm70_turbomind_fp8_linear(layer)


def test_apply_refuses_an_activation_whose_k_differs():
    layer = _layer()
    layer.sm70_fp8_in_features = 256
    kernels._check_activation_k(layer, torch.zeros(4, 256))
    with pytest.raises(RuntimeError, match="activation K disagrees"):
        kernels._check_activation_k(layer, torch.zeros(4, 128))


def test_apply_skips_the_check_when_no_weight_was_prepared():
    kernels._check_activation_k(torch.nn.Module(), torch.zeros(4, 128))


# --- MoE gemm shape ---------------------------------------------------------


def test_aligned_moe_shape_is_accepted():
    _check_gemm_shape(256, 256)
    _check_gemm_shape(384, 128, 128)


def test_moe_n_that_is_not_a_32_multiple_is_refused():
    with pytest.raises(RuntimeError, match="N a multiple of 32"):
        _check_gemm_shape(256, 104)


def test_moe_partial_k_group_is_refused():
    with pytest.raises(RuntimeError, match="quantization group"):
        _check_gemm_shape(192, 256)


def test_moe_empty_shape_is_refused():
    with pytest.raises(RuntimeError, match="positive K and N"):
        _check_gemm_shape(0, 256)


def _experts(k=256, n=256, group_size=128):
    return Experts(
        tm_weight=None,
        tm_scales=None,
        ptr_w=None,
        ptr_s=None,
        k_ld=k,
        q_ld=0,
        num_experts=2,
        k=k,
        n=n,
        group_size=group_size,
    )


def test_call_that_matches_the_prepared_experts_is_accepted():
    _check_call_matches_prepared(256, 256, _experts())


def test_call_whose_k_differs_from_the_prepared_experts_is_refused():
    with pytest.raises(RuntimeError, match="disagree with the prepared experts"):
        _check_call_matches_prepared(384, 256, _experts())
    with pytest.raises(RuntimeError, match="disagree with the prepared experts"):
        _check_call_matches_prepared(256, 128, _experts())


def test_gemm_validates_before_loading_the_binding():
    # A wrong K must fail without a TurboMind library in the environment.
    with pytest.raises(RuntimeError, match="disagree with the prepared experts"):
        gemm(None, None, None, _experts(), k=384, n=256)


def test_prepare_experts_refuses_a_bad_shape_before_loading():
    weights = torch.zeros(2, 104, 256, dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 1, 2)
    with pytest.raises(RuntimeError, match="N a multiple of 32"):
        prepare_experts(weights, scales)
