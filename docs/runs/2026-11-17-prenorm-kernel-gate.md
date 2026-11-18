# HC prenorm kernel: first gate run, and it misses

The gate was fixed before the kernel was written
(`docs/sm70-prenorm-kernel-design.md`). This is the first implementation's run
against it. **It misses at every shape**, so the kernel is not landed; the
numbers and the diagnosis are here so the next attempt starts from evidence.

**Arm.** Tesla V100-SXM2-32GB cap (7,0), torch 2.9.1+cu128, Triton 3.5.1,
CUDA 12.8, one card, no CUDA graphs, 10 warm-up iterations discarded, 50
repeats, median of per-iteration events. Kernel: two-launch deterministic
split-K (partials written, never atomically accumulated; a tiny reduce kernel
applies `rsqrt`), `SPLIT=8, BT=32, BK=128`, MIX padded to 32 for `tl.dot`.

| M | kernel ms | torch ms | kernel/torch | gate ms | gate met | rel err | two-run delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2243 | 0.1101 | 0.49x | 0.040 | no | 7.1e-7 | 0.0 |
| 4 | 0.2222 | 0.1126 | 0.51x | 0.040 | no | 9.8e-7 | 0.0 |
| 32 | 0.2232 | 0.1137 | 0.51x | 0.040 | no | 9.2e-7 | 0.0 |
| 256 | 0.2294 | 0.2202 | 0.96x | 0.106 | no | 1.1e-6 | 0.0 |
| 1024 | 0.6748 | 0.7485 | 1.11x | 0.369 | no | 1.3e-6 | 0.0 |
| 4096 | 2.0362 | 2.6061 | 1.28x | 1.000 | no | 1.2e-6 | 0.0 |

Agreement passes with room to spare: relative error is ~1e-6 against a 1e-2
gate, and the two-run delta is exactly zero, which confirms the design's
determinism claim (no atomics). Speed is the failure: the kernel is **slower
than torch below M=256** and only 1.28x faster at M=4096.

## Why, from the shapes

- At M <= 32 the grid is `ceil(T/32) x 8` = 8 programs on an 80-SM part, and
  each runs 20 serial iterations of a 32x128 x 128x32 fp32 dot. That is tiny
  tiles at 10% occupancy; the kernel spends its time reading `hc_fn` (1.97 MB)
  from 8 SMs instead of 80.
- `MIX=24` is padded to 32 for `tl.dot`, so a quarter of the arithmetic is
  padding at every size.
- fp32 has no tensor cores on sm70, so the GEMM is FMA-only regardless of the
  kernel language. cuBLAS on that path is highly tuned; a naive split-K Triton
  kernel is not a fair fight at large M either (1.28x is the honest margin).

The gate assumed a fused kernel could approach the 0.027 ms single-matmul
floor. That assumption is not refuted — this implementation does not reach it —
but the gap (8x at small M) is not a tuning slack, it is occupancy and tile
shape.

## What would be tried next, in order

1. Raise parallelism at small M: `SPLIT=40`, `BK=256`, `BT=16`, so M=1 still
   fills the part; keep the deterministic reduce.
2. Drop the 32-pad: 24 outputs as two dots of 16 and 8, or an explicit FMA
   accumulation loop instead of `tl.dot`.
3. Compare against a single-launch variant that reads `hc_fn` once with a full
   K loop, since at M <= 32 the cost is that read.

None of these are done, and none may be assumed to pass: the gate stays at the
numbers above, and a later run either meets them or reports the miss the same
way.

## What this does not say

- No bluewhale claim: torch fallback, synthetic tensors, one card.
- The kernel is not in the tree. It lives at lambda4
  `/tmp/mxfp4-verify/work/prenorm_kernel.py` until it passes, so nothing
  imports an unpassing kernel.
