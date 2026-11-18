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

## The batch band, measured against the second gate

After the first run, a second gate was declared for the small-batch band
(`>= 2x at B <= 8`, same tolerance and same top-64 equality). The band was then
measured with the same kernel and harness, captured, replay-only, 10 discards,
50 medians, card 4 (claim `port-31413`, closed):

| B | torch ms | kernel ms | ratio | >= 2x | max abs diff | top-64 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1403 | 0.0399 | 3.51x | yes | 1.2e-7 | yes |
| 2 | 0.1751 | 0.0707 | 2.48x | yes | 1.2e-7 | yes |
| 4 | 0.2647 | 0.1178 | 2.25x | yes | 1.2e-7 | yes |
| 8 | 0.4362 | 0.2232 | **1.95x** | **no** | 1.2e-7 | yes |
| 16 | 0.7199 | 0.3758 | 1.92x | no | 1.2e-7 | yes |
| 32 | 1.3414 | 0.6881 | 1.95x | no | 1.2e-7 | yes |

**The second gate is not met at its upper edge.** B=8 is 1.95x against a 2x bar,
and B=8 has now measured 2.04x and 1.95x in two runs a few minutes apart — a
4% run-to-run spread on a bar exactly at 2.0, so B=8 is not decidable by this
measurement. The band that does clear 2x unambiguously is **B <= 4** (2.25x at
B=4, 2.48 at B=2, 3.51 at B=1). Choosing between "declare the band at B<=4" and
"treat B=8 as inside the noise and declare B<=8" is the plan owner's call; this
record states the measurement, not the resolution.

At B=32 the same run puts the kernel at 0.6881 ms against torch's 1.3414, i.e.
51% of the call, so the eight index layers would go **10.74 -> 5.51 ms/step
(~5.2 ms/step saved)** if the large-batch arm were shipped. That is a single-op
inference from a synthetic call, not a served step measurement, and it is the
number a TileLang round should be weighed against.

## The wired arm, and its worth

The band edge was set at **B <= 4** and the kernel is wired as the shim's
`auto` arm for paged logits (component `paged_logits_sm70`,
`PAGED_LOGITS_AUTO_MAX_B = 4`). The edge is **noise-limited, not resolved**:
B=8 measured 2.04x and 1.95x on the same 2.0x bar, so a threshold there would be
a wash in half the runs; B=4 clears at 2.25x and the census's served arm
measured `m~4`.

**Two worth figures, each labelled by its arm, both single-op inferences from a
synthetic call rather than served numbers:**

- **B <= 4, wired:** 0.2647 -> 0.1178 ms per call, ~0.147 ms over the eight
  index layers, **~1.2 ms/step**.
- **B = 32, not wired:** 10.74 -> 5.51 ms/step, **~5.2 ms/step**, reachable only
  through a TileLang rewrite — the Triton kernel is 1.95x there, short of 3x.

If TileLang clears 3x at B=32, the two gates collapse, the crossover
disappears, and the dispatch should be retired in favour of a single arm.
