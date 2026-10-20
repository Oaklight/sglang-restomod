"""Large-row packed E4M3 GEMM using transient FP16 operands on Volta.

Unpack one projection once, then reuse it across all activation rows in cuBLAS.
The checkpoint remains packed; the caller owns the bounded projection scratch.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _fp16_output_module():
    return load_jit(
        "sm70_fp8_gemm_fp16_output",
        cuda_files=["gemm/sm70_fp8_gemm.cuh"],
        cuda_wrappers=[("run", "sm70_fp8_gemm_fp16_output")],
        extra_ldflags=["-lcublas"],
    )


@triton.jit
def _unpack_fp8(
    W,
    S,
    D,
    PN: tl.constexpr,
    PK: tl.constexpr,
    GROUP: tl.constexpr,
    MACRO: tl.constexpr,
    B: tl.constexpr,
):
    off = tl.program_id(0) * B + tl.arange(0, B)
    k, c = off // PN, off % PN
    tiles = MACRO // 64
    # Contiguous K-major stores outweigh the redundant packed-word reads.
    n = c
    n_tile = n // 64
    word = (
        (k // 16) * (PN * 4)
        + (n_tile // tiles) * tiles * 256
        + ((k % 16) * 16 + (n % 64) // 4) * tiles
        + n_tile % tiles
    )
    packed = tl.load(W + word, off < PN * PK, 0).to(tl.uint32)
    code = (packed >> (((n % 4) // 2 + 2 * (n % 2)) * 8)) & 255
    bits = ((code & 127) << 7) | ((code & 128) << 8)
    biased = bits.to(tl.uint16).to(tl.float16, bitcast=True)
    scale = tl.load(S + (k // GROUP) * PN + n, off < PN * PK, 0)
    value = (biased.to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
    tl.store(D + k * PN + n, value, off < PN * PK)


def sm70_fp8_gemm(x, weight, scales, output, scratch):
    """Packed FP8 / FP16-scale GEMM with full-K FP32 accumulation.

    Scratch is FP16 [physical_K, physical_N], used only during this call.
    Scales are [physical_K/group, physical_N], with group32 or128. Logical
    widths may be padded. Wide projections disable reduced-precision reduction
    on the cuBLAS handle; skinny projections use a FP32 output temporary to
    retain the faster sliced-K algorithm. Both round only the final output.
    """
    rows, k = x.shape
    groups, pn = scales.shape
    pk = weight.shape[0] * 16
    n = output.shape[1]
    group = pk // groups
    if (
        x.device.type != "cuda"
        or torch.cuda.get_device_capability(x.device) != (7, 0)
        or x.dtype != torch.float16
        or weight.dtype != torch.int32
        or scales.dtype != torch.float16
        or output.dtype != torch.float16
        or scratch.dtype != torch.float16
        or group not in (32, 128)
        or pk != groups * group
        or pk % 128
        or pn % 128
        or not 0 < k <= pk
        or not 0 < n <= pn
        or weight.shape != (pk // 16, 4 * pn)
        or output.shape != (rows, n)
        or scratch.shape != (pk, pn)
        or torch.backends.cuda.matmul.allow_fp16_accumulation
    ):
        raise ValueError("Unsupported SM70 packed FP8 GEMM layout or accumulation mode")
    if any(
        t.device != x.device or not t.is_contiguous()
        for t in (x, weight, scales, output, scratch)
    ):
        raise ValueError("SM70 FP8 GEMM requires contiguous colocated tensors")
    _unpack_fp8[(triton.cdiv(pk * pn, 1024),)](
        weight,
        scales,
        scratch,
        pn,
        pk,
        group,
        256 if pn % 256 == 0 else 128,
        1024,
        enable_fp_fusion=False,
    )
    if pn >= 2048:
        with torch.cuda.device(x.device):
            _fp16_output_module().run(
                x, scratch, output, torch.cuda.current_blas_handle()
            )
    else:
        output.copy_(torch.mm(x, scratch[:k, :n], out_dtype=torch.float32))
    return output
