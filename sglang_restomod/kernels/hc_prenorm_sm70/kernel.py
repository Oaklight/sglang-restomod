"""The Triton body of the fused HC prenorm: deterministic split-K, two launches.

Launch 1 has a grid of ``(ceil(T / BT), splits)``. Each program streams one K
slice of ``x`` and of ``fn``, accumulating both the ``[BT, MIX]`` dot products
and the ``[BT]`` row sums of squares, and writes them to per-split buffers.
Launch 2 sums the splits and writes ``out`` and ``sqrsum``.

Nothing is accumulated atomically, so the result does not depend on the order in
which blocks finish: two runs are bit-identical by construction, which the gate
measured at delta 0.0. The cost is one extra buffer and one extra launch.

``fn`` is fp32 and ``x`` is bf16/fp16/fp32 as it arrives; both are widened to
fp32 in registers, matching the reference's ``x.float()`` and ``fn.float()``
without materializing either cast.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BT = 32
BK = 128


@triton.jit
def _partial_kernel(x_ptr, fn_ptr, part_ptr, ss_ptr, T, HC, MIX,
                    SPLIT: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
                    MP: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_t * BT + tl.arange(0, BT)
    row_mask = rows < T
    mix = tl.arange(0, MP)
    mix_mask = mix < MIX

    k_per_split = HC // SPLIT
    k_end = pid_s * k_per_split + k_per_split

    acc = tl.zeros((BT, MP), dtype=tl.float32)
    ss = tl.zeros((BT,), dtype=tl.float32)
    for kk in range(0, k_per_split, BK):
        cols = pid_s * k_per_split + kk + tl.arange(0, BK)
        # A split whose slice is not a multiple of BK has a ragged tail; masking
        # it keeps the load in bounds rather than reading the next split's data.
        col_mask = cols < k_end
        x_blk = tl.load(x_ptr + rows[:, None] * HC + cols[None, :],
                        mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        fn_blk = tl.load(fn_ptr + mix[:, None] * HC + cols[None, :],
                         mask=mix_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(x_blk, tl.trans(fn_blk))
        ss += tl.sum(x_blk * x_blk, axis=1)

    tl.store(part_ptr + pid_s * T * MIX + rows[:, None] * MIX + mix[None, :], acc,
             mask=row_mask[:, None] & mix_mask[None, :])
    tl.store(ss_ptr + pid_s * T + rows, ss, mask=row_mask)


@triton.jit
def _reduce_kernel(part_ptr, ss_ptr, out_ptr, sq_ptr, T, MIX,
                   SPLIT: tl.constexpr, BT: tl.constexpr, MP: tl.constexpr):
    pid_t = tl.program_id(0)
    rows = pid_t * BT + tl.arange(0, BT)
    row_mask = rows < T
    mix = tl.arange(0, MP)
    mix_mask = mix < MIX

    acc = tl.zeros((BT, MP), dtype=tl.float32)
    ss = tl.zeros((BT,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * T * MIX + rows[:, None] * MIX + mix[None, :],
                       mask=row_mask[:, None] & mix_mask[None, :], other=0.0)
        ss += tl.load(ss_ptr + s * T + rows, mask=row_mask, other=0.0)

    tl.store(out_ptr + rows[:, None] * MIX + mix[None, :], acc,
             mask=row_mask[:, None] & mix_mask[None, :])
    tl.store(sq_ptr + rows, ss, mask=row_mask)


def launch(x, fn, out, sqrsum, splits: int) -> None:
    """Run both launches. Shapes and dtypes are already validated by the caller."""
    t, hc = x.shape
    mix = fn.shape[0]
    mp = max(16, triton.next_power_of_2(mix))
    part = torch.empty(splits, t, mix, dtype=torch.float32, device=x.device)
    ss = torch.empty(splits, t, dtype=torch.float32, device=x.device)
    grid_t = triton.cdiv(t, BT)
    _partial_kernel[(grid_t, splits)](
        x, fn, part, ss, t, hc, mix,
        SPLIT=splits, BT=BT, BK=BK, MP=mp)
    _reduce_kernel[(grid_t,)](
        part, ss, out, sqrsum, t, mix,
        SPLIT=splits, BT=BT, MP=mp)
