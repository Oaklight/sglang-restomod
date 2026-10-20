"""Small-row E4M3 GEMV over the packed SM70 Marlin layout.

Reconstruct the same FP16 operands as Marlin and reduce FP32 partials before
the FP16 output boundary. Weights remain packed; scratch belongs to the caller.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_partial(
    X,
    W,
    S,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    PN: tl.constexpr,
    PK: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MACRO: tl.constexpr,
    ROWS: tl.constexpr,
):
    lane = tl.arange(0, BN)
    tiles = MACRO // 64
    macro = tl.program_id(0) // (MACRO // BN)
    sub = tl.program_id(0) % (MACRO // BN)
    n = macro * MACRO + (lane % tiles) * 64 + sub * (BN // tiles) + lane // tiles
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    n_tile = n // 64
    word = (
        (k[:, None] // 16) * (PN * 4)
        + (n_tile[None, :] // tiles) * tiles * 256
        + ((k[:, None] % 16) * 16 + (n[None, :] % 64) // 4) * tiles
        + n_tile[None, :] % tiles
    )
    packed = tl.load(W + word).to(tl.uint32)
    shift = ((n % 4) // 2 + 2 * (n % 2)) * 8
    code = (packed >> shift[None, :]) & 255
    bits = ((code & 127) << 7) | ((code & 128) << 8)
    biased = bits.to(tl.uint16).to(tl.float16, bitcast=True)
    scales = tl.load(S + (k[:, None] // GROUP) * PN + n[None, :])
    weight = (biased.to(tl.float32) * scales.to(tl.float32)).to(tl.float16)
    # Reuse each reconstructed weight across the small verification batch.
    for row in tl.static_range(ROWS):
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
        accum = tl.sum(x[:, None] * weight.to(tl.float32), axis=0)
        tl.store(P + (row * (PK // BK) + tl.program_id(1)) * N + n, accum, n < N)


@triton.jit
def _fp8_reduce(
    P,
    Y,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    s = tl.arange(0, BS)
    partial = tl.load(
        P + (row * SPLITS + s[:, None]) * N + n[None, :],
        (s[:, None] < SPLITS) & (n[None, :] < N),
        0,
    )
    tl.store(Y + row * N + n, tl.sum(partial, axis=0), n < N)


def sm70_fp8_gemv(x, weight, scales, output, partials):
    """Apply packed E4M3/FP16-scale weights to one through four FP16 rows.

    Weight shape is [PK/16,4*PN], scales [PK/group,PN], with group32 or128.
    Logical input/output widths may be smaller than the padded physical widths.
    Caller-owned scratch is FP32 [rows,PK/128,logical_N].
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
        or partials.dtype != torch.float32
        or not 1 <= rows <= 4
        or group not in (32, 128)
        or pk != groups * group
        or pk % 128
        or pn % 128
        or not 0 < k <= pk
        or not 0 < n <= pn
        or weight.shape != (pk // 16, 4 * pn)
        or output.shape != (rows, n)
        or partials.shape != (rows, pk // 128, n)
    ):
        raise ValueError("Unsupported SM70 packed FP8 GEMV layout")
    if any(
        t.device != x.device or not t.is_contiguous()
        for t in (x, weight, scales, output, partials)
    ):
        raise ValueError("SM70 FP8 GEMV requires contiguous colocated tensors")
    macro = 256 if pn % 256 == 0 else 128
    _fp8_partial[(pn // 64, pk // 128)](
        x,
        weight,
        scales,
        partials,
        n,
        k,
        pn,
        pk,
        group,
        64,
        128,
        macro,
        rows,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _fp8_reduce[(triton.cdiv(n, 128), rows)](
        partials,
        output,
        n,
        pk // 128,
        triton.next_power_of_2(pk // 128),
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
