# Paged MQA logits kernel: agreement passes, speed misses (1.94x of 3x)

First implementation against the B=32 gate in `docs/sm70-indexer-kernel-design.md`.
**Rejected on speed**, reported rather than retuned, and not wired into the shim.
The kernel is not in the tree; it lives at lambda4
`/nfs/ml_lab/projects/ml_lab/pding/scratch/indexer-kernel/paged_logits.py`.

**Arm.** Tesla V100-SXM2-32GB cap (7,0), torch 2.9.1+cu128, CUDA 12.8, card 4
(claim `port-31412`, closed), Triton 3.5.1. Both arms captured verbatim in one
`torch.cuda.CUDAGraph` each, allocations inside, 10 warm-up replays discarded,
50 timed, median, **replay only** — kernel-crafter's `indexer_bench.py` with one
import changed, so the arm is the one the baseline was priced in.

| B | pages/seq | torch ms | kernel ms | ratio | gate | max abs diff | top-64 identical |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 1.3373 | 0.7347 | **1.94x** | >= 3x (<= 0.447) | 1.2e-7 | yes |
| 8 | 128 | 0.3871 | 0.1905 | 2.04x | — | 1.2e-7 | yes |
| 1 | 128 | 0.1234 | 0.0348 | **3.53x** | — | 1.2e-7 | yes |

Agreement is not close: 1.2e-7 against a 2e-2 tolerance, and the top-64 set is
identical on **every row** at all three shapes. The kernel is one launch, reads
each KV byte once, and materializes neither the bf16 cast nor the fp32 score
tensor. It is simply not fast enough at the served batch.

## Why, and it is the dot, not the memory

2.15 GFLOP in 0.73 ms is **2.9 TFLOP/s**, about 18% of the part's ~15.7 fp32
TFLOPS. The unavoidable read (34.6 MB gathered) has a 0.049 ms floor at
700 GB/s, so the kernel is nowhere near memory-bound either. Measured variants,
eager, B=32, same inputs (50 medians):

| variant | ms |
| --- | --- |
| fp32 dot, `num_warps=8`, decode in chunks of 32 dims | **0.86** |
| fp32 dot, `num_warps=4`, chunks of 32 / 64 | 0.89 / 0.93 |
| fp32 dot, `num_warps=16` | 1.53 |
| fp16 dot (e4m3->fp16 is exact), `num_warps=8` | 0.87 |
| bf16 dot, `num_warps=4` / 8 / 16 | **9.16 / 9.93 / 9.48** |
| full 128-wide decode in one tile, fp32 dot, 8 warps | 0.98 |

Two facts follow. **Triton on sm70 has no tensor-core path for this dot**: fp16
and fp32 land within noise of each other, which is the FMA lowering both times.
And **the bf16 dot is pathological** — 10x slower than fp32 on this version, so
the design's "16-bit MMA on sm70" assumes an MMA lowering that Triton does not
produce. Chunking the decode moved the total by 8%, so the e4m3 decode is not
the bottleneck; the dot is.

## What would change the answer

The op needs a real MMA path, and Triton does not emit one on sm70
(`Triton 3.5.1 emits no tensor-core ops on sm70`). TileLang does, and e4m3->fp16
is exact, so the operands survive a 16-bit MMA unchanged. That is the indicated
next attempt if this op is revisited.

There is also a cheaper, narrower option the numbers already support: at B=1 the
kernel is **3.53x**, because the torch arm still carries seven launches there.
A B<=1-only dispatch would pass the gate's ratio on its own arm and take 1.12 ->
0.35 ms/step of the 8-layer cost. That is a decision for the plan, not a
retune — the gate is aimed at B=32, where the money is (10.74 ms/step).

## What this does not say

- No bluewhale claim: the baseline is the torch fallback on synthetic tensors.
- No claim about the CUDA flash_mla path's numbers, only the fallback's.
- Nothing about the shim: `fp8_paged_mqa_logits` still routes to the torch twin,
  and no `fused` arm was added, since nothing passed.
