# QPN2 functional guard (P2 fixture; P3/P4 cross-run)

Purpose: catch a **shared** misreading between the QPN2 spec and the repack by
comparing model outputs, not weights. A byte cross-check against 1Cat would
catch a difference; it cannot catch both implementations agreeing on a wrong
lane/K order. The current Marlin path is an independent, known-good consumer of
the same checkpoint, so output equality between the two paths is the functional
check.

## 0. Memory constraint: no dual-layout guard on the production model

Both layouts are 0.5625 B/weight. P0 established that the production NVFP4
checkpoint is ~126 GB and, at TP4, each layout is 29.6-30.0 GB per card. Holding
Marlin and QPN2 at once would be ~60 GB/card against 32 GB, so a single process
**cannot** run the arm matrix on the production checkpoint. The guard is split:

- **P2**: decode-only, on a small fixture where both layouts fit, in one
  process. Isolates the decode kernel and localises failures.
- **P3/P4**: full replacement, two separate runs, each holding one layout.
  Compares outputs across runs; no process holds both.

Do not attempt the dual-layout matrix on the real model.

## 1. P2: decode-only guard on a small fixture (one process)

Fixture, both resident:

- an 8-expert NVFP4 bank with the production per-rank shapes: `w13` N=320 /
  K=2560 and `w2` N=2560 / K=160. Each layout is ~5.5 MB, so both fit with no
  pressure;
- optionally a reduced NVFP4 MoE model on the same shapes for model-level arms.

Switch via `SGLANG_SM70_FP4_MOE_LAYOUT=marlin|qpn2` and
`SGLANG_SM70_QPN2_TENSORS=both|w13|w2`. Only the decode-shaped expert GEMMs
(`M <= 8`) switch; attention, shared experts, router, norms and LM head are
identical in both arms. Prefill is not part of P2.

Arms, all in one process:

1. Marlin baseline;
2. QPN2 both;
3. QPN2 w13 only;
4. QPN2 w2 only;
5. MTP-off arm if the fixture has a draft head.

At the kernel level the arms compare the expert-bank outputs directly (same
input, max relative error `<= 1e-3`). At the model level they compare token IDs
and logprobs with the rules in section 4.

## 2. P3/P4: full replacement, two runs, no dual layout

Two separate launches, same checkpoint, same 4-card island, same flags, same
prompts; each process holds exactly one layout:

- **run A**: the QPN2 branch's parent commit (pre-QPN2), Marlin everywhere;
- **run B**: the QPN2 branch head, QPN2 everywhere including prefill.

The commit difference is the QPN2 change only, so the comparison remains
single-cause. The env flag stays as the switch until the prefill reader lands,
but the production guard is the cross-run comparison, not a dual-layout process.

**Run the primary cross-run arm with MTP off.** Rejection sampling is stochastic
even at `temperature 0` (issue #217: 3.20 against 3.37 accept length over six
reps on a fixed prompt), so token equality across separate runs cannot be
attributed to the layout while MTP is on. With MTP off, greedy decoding is
deterministic and the only difference is the expert path. MTP-on is a secondary
check on accept length and logprobs, not on token equality.

## 3. Prompt set

Fixed, greedy, batch size 1 (`--max-running-requests 1`), 128 output tokens.

| axis | values |
|---|---|
| context | 76, 993, 3,473, 7,168 tokens |
| content | (a) repeated technical seed from `scripts/bench.py`; (b) unique RNG word prompt, fixed seed per context |
| prompts | 2 per (context, content) |

16 prompts. One warmup request per prompt (JIT/capture) is discarded, then 2
reps. The 7,168-token cell is the long-context/address case; 76 is the shortest
decode; 993/3,473 are the recipe rows. For a reduced fixture, keep the token
counts where the fixture's context allows and drop the cells it cannot reach,
but keep at least one long cell.

Record per prompt: full 128 greedy token IDs, per-token logprob of the chosen
token and the top-2 logprobs at each position, accept length and verify count.

## 4. Tolerance and pass/fail

Both arms consume the same weights, so the only expected difference is
accumulation order. A layout error permutes K or columns and changes the expert
output by O(1).

- **Pass**: identical 128-token greedy sequences on all prompts and reps, and
  max per-token `|delta logprob| <= 1e-3` nats.
- **Near-tie pass**: a mismatch is accepted only when the Marlin top-2 logprob
  gap at that position is `< 1e-3` nats and the QPN2 token is the second
  choice. No mismatch is allowed in the first 8 generated tokens.
- **Fail**: any mismatch with `|delta logprob| > 1e-3`; systematic shift,
  repetition or truncation; a crash; or a per-tensor arm failing while `both`
  passes.
- **Draft check** (fixture or P3/P4 with MTP on): accept length within 0.05 of
  Marlin. A larger drop with matching target tokens localizes to the draft.

## 5. What a failure localizes

| observation | localization |
|---|---|
| `w13`-only fails, `w2`-only passes | gate_up layout: tile/lane mapping (N=320, tiles=10) |
| `w2`-only fails, `w13` passes | down layout: group split/stride (K=160, groups=10) |
| both fail at token 1 | first decode call; check w13/w2 addressing and the expert stride |
| tokens match, logprobs differ by a constant scale | scale path: fp16 RNE vs float32, or the global-scale multiply |
| divergence only at the longest context | address overflow / padded or last-tile handling |
| divergence grows with position | accumulation or a per-layer stride error; bisect the layer if a layer toggle is added |
| MTP accept drops, target tokens match | draft path uses the same QPN2 expert GEMM |

Record the first divergent prompt, position, the two top-2 logprob vectors and
the token IDs.

## 6. Cost and order

1. P2 single-layer expert-bank check (`M = 1..4`, seconds, one card);
2. P2 arm matrix on the fixture: one process, both layouts, the four arms;
3. P3/P4: run A (parent, Marlin) and run B (QPN2 branch) on the production
   checkpoint, MTP off for the primary comparison (~4 min load each + requests);
4. only on failure, `w13`-only and `w2`-only runs if the fixture did not
   already localize the tensor.

Acceptance evidence: the token/logprob table, accept lengths, and the
first-divergence artifact, recorded in the lab note for P2 (fixture) and P3/P4
(full replacement).
