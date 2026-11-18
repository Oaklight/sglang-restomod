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

## Second run: both arms inside a captured CUDA graph

The first gate compared uncaptured timings, but decode is graph-captured in the
serving configuration, so most of torch's 0.11 ms was inter-launch gap — the
part a graph removes and a two-launch kernel already had less of. This run
captures each arm **verbatim**, allocations included (in serving
`hc_pre_torch_impl` runs inside the captured forward, so its allocations are
captured too), and times **replay only**: a graph launch is inside every
replay, nothing runs in python outside it.

**Arm.** Same card and stack: Tesla V100-SXM2-32GB cap (7,0), torch 2.9.1+cu128,
Triton 3.5.1, one graph per arm via `torch.cuda.CUDAGraph().capture` on a side
stream after 3 warm-up calls, 10 replays discarded, 50 replays timed, median,
`kernel` geometry unchanged (SPLIT=8, BT=32, BK=128).

| M | torch eager | torch graph | kernel eager | kernel graph | kernel/torch (graph) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1137 | **0.0266** | 0.2232 | 0.1894 | 7.1x slower |
| 4 | 0.1167 | **0.0328** | 0.2232 | 0.1894 | 5.8x slower |
| 32 | 0.1157 | **0.0461** | 0.2232 | 0.1905 | 4.1x slower |
| 256 | 0.2212 | 0.1997 | 0.2304 | 0.1956 | 1.02x faster (tie) |

Determinism survives capture: both arms give a replay-to-replay delta of
exactly 0.0, and the kernel's agreement is unchanged at 7.1e-7 to 1.1e-6.
`torch_graph_rel_err` is 0 by construction (eager reference is the same code).

## Decision

The rule fixed before this run: if torch falls to ~0.06 ms under capture, the
kernel is not within reach at decode and the answer is a shape-gated dispatch.
Torch falls to 0.0266 ms — well under it. **The kernel stops here**, with no
geometry iteration. What the two runs together support:

- **Decode (in-graph, M = 1-32): torch wins by 4-7x.** The kernel's 0.19 ms is
  real GPU work on 8 CTAs, not launch gap; capture shrank it only 15%.
- **Large M, eager (prefill): the kernel wins 1.11x at 1024 and 1.28x at 4096.**
  Prefill is not graph-captured, so the first run's eager numbers are the right
  comparison there.
- **Crossover is around M=256**, where the two tie. That is the shape gate
  upstream already uses.

## Correction to the baseline this work was scoped against

The 4.7 ms/step decode figure is `0.117 x 40` measured **without** graphs. In
the graph-captured regime the same 40 calls cost 1.06-1.85 ms/step
(0.027-0.046 x 40), so the prenorm's decode cost is roughly a quarter of what
the scoping number said. The remaining floor is data, not launches: 40 layers
x 1.97 MB of `hc_fn` = 79 MB per step, which at V100 bandwidth is ~0.1 ms. A
win that large needs one kernel reading many layers' `hc_fn` at once — fewer
calls, per the coordinator's note — not a faster single-layer kernel.

## Third run: the shipped kernel against the arm the engine actually calls

The first two runs compared against a harness `torch_ref` that carries 6-7
launches (cast, square, mean, add, rsqrt, linear, mul). The arm the engine
reaches is the shim's `_hc_prenorm_reference`: cast, `mm`, mul+sum -- 3
launches, writing into caller-allocated `out` and `sqrsum`. That is a lower bar,
so it is the one the shipped kernel has to beat. Eager, replay not applicable
(prefill is not graph-captured), 10 warm-up, 50 reps, median, kernel as shipped
in `sglang_restomod/kernels/hc_prenorm_sm70`.

| M | reference ms | kernel ms | kernel/reference | rel err (out) | rel err (sqrsum) |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.2089 | 0.2499 | **0.84x** | 1.1e-6 | 1.8e-7 |
| 1024 | 0.7444 | 0.7107 | **1.05x** | 1.3e-6 | 1.8e-7 |
| 4096 | 2.6378 | 1.9932 | **1.32x** | 1.2e-6 | 1.8e-7 |
| 8192 | 5.1610 | 3.6511 | **1.41x** | 1.4e-6 | 2.8e-7 |

So the prefill win survives against the leaner arm, and shrinks less than the
launch-count argument predicted: at M=4096 the reference is 2.64 ms for three
launches, i.e. its cost was never mostly launches either, it was the fp32 cast
of a 335 MB tensor plus the square-and-sum over it. The kernel's gain is
removing that intermediate, not saving launches.

That fixes the auto threshold: below 256 the kernel loses, at 1024 it is a
tie, and it leads from there. `HC_PRENORM_AUTO_MIN_M = 1024` is the right
line, and at M=1024 the cost of choosing the kernel is measurable but ~5%.
