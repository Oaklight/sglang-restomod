# The indexer fallback does not collapse under graph capture

The prenorm's eager 0.114 ms fell to 0.027 inside a captured graph, which
reframed that op's budget by 4x. The indexer was priced the same way — 8 x
0.287 = 2.3 ms/step from an eager measurement — so it was measured the same
way before any kernel work. **It behaves differently**: capture removes about
half the time at batch 1 and about a tenth at batch 8 and above, and the cost
then scales with batch and pages rather than sitting flat.

**Arm.** Tesla V100-SXM2-32GB cap (7,0), torch 2.9.1+cu128, CUDA 12.8, card 4,
one `torch.cuda.CUDAGraph` per shape captured verbatim on a side stream after 3
warm-up calls (allocations inside, as at serving), 10 replays discarded, 50
timed, median, **replay only**. Shapes: `H=32`, `head_dim=128`, `block_size=64`,
random `seq_lens`-masked page table over the whole cache, `clean_logits=False`.
Call is `fp8_paged_mqa_logits_torch` from the site tree, unmodified.

| B | pages/seq | seq_len | eager ms | graph ms | graph/eager | x8 layers |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 8192 | 0.2883 | 0.1403 | 0.49 | 1.12 ms/step |
| 8 | 128 | 8192 | 0.4772 | 0.4332 | 0.91 | 3.47 ms/step |
| 32 | 128 | 8192 | 1.4781 | 1.3425 | 0.91 | 10.74 ms/step |
| 1 | 2048 | 131072 | 0.7670 | 0.6984 | 0.91 | 5.59 ms/step |

Determinism and fidelity under capture are exact: replay-to-replay delta 0.0 and
graph output bit-identical to the eager output in every row (`graph_matches_eager`
0.0), which is expected for a chain of deterministic kernels with no atomics.

## What it changes

- **The business case is smaller at batch 1 and larger at batch.** At B=1 the
  in-graph cost is 1.12 ms/step, not 2.3. At B=32 it is 10.7 ms/step — the
  larger term in the step of anything this work has looked at, and it is real
  work, not launch gap.
- **"Flat from 2048 to 8192 pages" does not generalize.** At fixed B=1 the cost
  goes 0.1403 -> 0.6984 ms as pages go 128 -> 2048, i.e. roughly with the
  gathered volume. The earlier flatness was measured over a narrow range at one
  batch size.
- **The op is far from both limits.** At B=32, 1.34 ms moves 32 x 128 x 8448 B
  = 34.6 MB of gathered KV (~26 GB/s) and 2.15 GFLOP of bf16 dot
  (~1.6 TFLOP/s), against a V100's ~700 GB/s and ~28 TFLOP/s. The intermediate
  bf16/fp32 tensors (67 MB cast, 33.5 MB of scores, rewritten several times)
  dominate: it is a chain of full-size elementwise passes, which is what a
  fused kernel removes.

## What this does not say

- The prenorm conclusion does not transfer: this op is not overhead-bound once
  B > 1, so the argument "a graph already fixed it" fails here.
- No served measurement. These are synthetic single-op timings; a serving step
  interleaves these calls with attention, and the fused kernel's real win has to
  come from a served A/B.
- The gate in `docs/sm70-indexer-kernel-design.md` is unchanged and still
  untested; this only re-prices the baseline it is judged against.
