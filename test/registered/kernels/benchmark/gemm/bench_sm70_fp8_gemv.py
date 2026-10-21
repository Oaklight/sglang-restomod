"""Cold-cache packed FP8 GEMV versus a resident FP16 reference projection."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.sm70_fp8_gemv import sm70_fp8_gemv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _fp16(x, weight, output):
    output.copy_(torch.mm(x, weight.T, out_dtype=torch.float32))
    return output


@marker.parametrize(
    "n,k,rows",
    [(3072, 4096, 1), (3072, 4096, 4), (4096, 256, 1), (4096, 256, 4), (4096, 1024, 4)],
    [(3072, 4096, 4), (4096, 256, 1)],
)
@marker.benchmark("provider", ["packed", "fp16"])
def benchmark(n, k, rows, provider):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        return marker.skip("Requires SM70 packed FP8 kernels")
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    output = torch.empty(rows, n, device="cuda", dtype=torch.float16)
    values = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    if provider == "fp16":
        weight = (values.float() / 1024).half()
        return marker.do_bench(_fp16, input_args=(x, weight, output))

    # Set each encoded scale to1/4: the E4M3 exponent-bias convention adds2**8.
    # Prepare the Marlin layout outside timing; no extension/plugin is needed.
    scales = torch.full((k // 32, n), 0.25, device="cuda", dtype=torch.float16)
    kk = torch.arange(k, device="cuda")[:, None]
    nn = torch.arange(n, device="cuda")[None, :]
    word = (kk // 16) * (n * 4) + (nn // 256) * 1024
    word = word + ((kk % 16) * 16 + (nn % 64) // 4) * 4 + (nn // 64) % 4
    shift = ((nn % 4) // 2 + 2 * (nn % 2)) * 8
    packed = torch.zeros(k * n // 4, device="cuda", dtype=torch.int64)
    packed.scatter_add_(
        0, word.flatten(), (values.view(torch.uint8).T.long() << shift).flatten()
    )
    weight = packed.int().reshape(k // 16, 4 * n)
    partials = torch.empty(rows, k // 128, n, device="cuda", dtype=torch.float32)
    # Default cloning rotates all read/write buffers past L2, including scratch.
    return marker.do_bench(
        sm70_fp8_gemv, input_args=(x, weight, scales, output, partials)
    )


if __name__ == "__main__":
    benchmark.run()
