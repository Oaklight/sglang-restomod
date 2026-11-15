"""GPU tests for the draft NVFP4 quantizer.

The expert GEMM itself is not re-tested here: a converted layer produces the
same checkpoint-layout triple a real NVFP4 checkpoint does, and
``test_sm70_fp4_moe_gpu.py`` already drives that triple through
``prepare_moe_nvfp4`` and the Marlin kernel. What is new is the quantizer, so
these tests pin that it computes the same thing on the device as on the CPU and
that the engine's reader agrees with it there too.

The end-to-end check is the served arm's accept rate, which no unit test can
stand in for.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")


def _device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def test_device_result_matches_cpu_bit_for_bit():
    from sglang_restomod.quant.sm70_mtp_fp4_moe.quantize import quantize_experts

    torch.manual_seed(0)
    bank = torch.randn(4, 16, 128, dtype=torch.float16)

    on_cpu = quantize_experts(bank)
    on_gpu = quantize_experts(bank.to(_device()))

    assert torch.equal(on_gpu.weight.cpu(), on_cpu.weight)
    assert torch.equal(on_gpu.scales.cpu().view(torch.uint8), on_cpu.scales.view(torch.uint8))
    assert torch.equal(on_gpu.global_scale.cpu(), on_cpu.global_scale)


def test_engine_reader_round_trips_on_the_device():
    from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

    from sglang_restomod.quant.sm70_mtp_fp4_moe.quantize import quantize_experts

    torch.manual_seed(0)
    device = _device()
    bank = torch.randn(4, 16, 128, dtype=torch.float32, device=device)

    packed = quantize_experts(bank)
    restored = dequantize_nvfp4(
        packed.weight,
        packed.scales,
        packed.global_scale.view(-1, 1, 1),
        out_dtype=torch.float32,
    )

    block_amax = bank.view(4, 16, 8, 16).abs().amax(dim=-1)
    limit = (block_amax / 6.0 * 1.15).repeat_interleave(16, dim=-1)
    assert bool(((restored - bank).abs() <= limit + 1e-6).all())
