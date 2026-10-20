import sys
from types import SimpleNamespace

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_global_scale,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


@pytest.fixture
def sm70_runtime():
    from sglang.srt.runtime_context import create_context, use_context
    from sglang.srt.server_args import ServerArgs

    context = create_context(
        ServerArgs(model_path="dummy", disable_overlap_schedule=True), role="test"
    )
    with use_context(context):
        yield


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Requires the SM70 plugin and group32 Marlin extension",
)
@pytest.mark.parametrize("n,k", [(160, 160), (1024, 4096), (19360, 4096)])
def test_sm70_mxfp8_padded_dense_bias_and_graph(sm70_runtime, n, k):
    """Group32 scales must not collapse to128; padding must not shift bias/head rows."""
    from sglang_v100_plus.fp8 import SM70MXFP8LinearMethod

    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.plugins import load_plugins

    load_plugins()
    torch.manual_seed(402)
    method = SM70MXFP8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            use_mxfp8=True,
        )
    )
    layer = torch.nn.Module()
    layer.tp_group = None
    with torch.device("cuda"):
        method.create_weights(layer, k, [n], k, n, torch.float16)
    bits = torch.randint(254, (n, k), device="cuda", dtype=torch.int32)
    bits = (bits + (bits >= 127)).byte()
    weight = bits.view(torch.float8_e4m3fn)
    scales = (
        (102 + torch.arange(n * (k // 32), device="cuda") % 21).byte().reshape(n, -1)
    )
    layer.weight.data.copy_(weight)
    layer.weight_scale_inv.data.copy_(scales)
    reference = (
        weight.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
    ).half()
    method.process_weights_after_loading(layer)
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
    for rows in (1, 4, 33, 257):
        x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
        method.apply(layer, x, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = method.apply(layer, x)
            biased = method.apply(layer, x, bias)
        x.normal_(std=0.1)
        bias.normal_(std=0.01)
        graph.replay()
        # Independent FP64 dot product, with the required FP16 operand/output
        # and post-projection bias boundaries retained explicitly.
        expected = (x.double() @ reference.double().T).half()
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.0002)
        # Check the two rounding boundaries separately. Composing half output
        # rounding with half bias rounding can turn a one-ULP dot difference
        # into two ULPs; that does not imply an incorrect bias epilogue.
        torch.testing.assert_close(biased, actual.add(bias), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Requires SM70 NVFP4 Marlin",
)
def test_sm70_nvfp4_bias_after_padded_output_crop(sm70_runtime):
    """Vision's logical bias cannot broadcast against Marlin's padded N output."""
    from sglang.srt.plugins import load_plugins

    load_plugins()
    torch.manual_seed(403)
    n, k = 96, 128
    layer = torch.nn.Module()
    layer.params_dtype = torch.float16
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.input_size_per_partition, layer.output_size_per_partition = k, n
    codes = torch.randint(16, (n, k), device="cuda", dtype=torch.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = torch.full((n, k // 16), 0.25, device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    for name, value in [
        ("weight", packed),
        ("weight_scale", scales),
        ("weight_global_scale", torch.tensor(0.01, device="cuda")),
        ("bias", torch.randn(n, device="cuda", dtype=torch.float16) * 0.01),
    ]:
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    reconstructed = (
        lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1, 1) * 0.25 * 0.01
    ).half()
    prepare_nvfp4_layer_for_marlin(layer)
    x = torch.randn(3, k, device="cuda", dtype=torch.float16) * 0.1
    actual = apply_fp4_marlin_linear(
        x,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        n,
        k,
        layer.bias,
    )
    expected = (x.double() @ reconstructed.double().T).half().add(layer.bias)
    torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.0002)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin fallback tests require CUDA SM8X/SM9X/SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_support_and_scale_transforms_sm80_sm90_sm120(dtype):
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    assert check_marlin_supported(
        scalar_types.float4_e2m1f,
        group_size=16,
        has_zp=False,
        device_capability=capability,
    )

    global_scale = torch.tensor(1.0, dtype=dtype, device="cuda")
    actual_global_scale = nvfp4_marlin_process_global_scale(global_scale)
    assert actual_global_scale.is_cuda
    assert actual_global_scale.ndim == 1
    assert actual_global_scale.numel() == 1
    if dtype == torch.float16:
        assert actual_global_scale.item() == 128.0
    else:
        assert actual_global_scale.item() == 2.0**119


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="NVFP4 Marlin dense numeric test requires CUDA SM80, SM86, SM90 or SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_marlin_dense_matches_dequant_reference(dtype):
    torch.manual_seed(0)

    size_m = 17
    size_k = 256
    size_n = 192
    group_size = 16

    a_input = torch.randn((size_m, size_k), dtype=dtype, device="cuda") / 10
    fp4_weight, scales, global_scale, weight_ref = make_nvfp4_weight_and_ref(
        size_n, size_k, dtype, group_size=group_size
    )

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.params_dtype = dtype
    layer.weight = torch.nn.Parameter(fp4_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        global_scale.reshape(1), requires_grad=False
    )
    prepare_nvfp4_layer_for_marlin(layer)

    output = apply_fp4_marlin_linear(
        a_input,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        size_n,
        size_k,
        use_fp32_reduce=True,
    )

    output_ref = torch.matmul(a_input, weight_ref.T)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
