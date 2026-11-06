# QPN2 functional guard (P2 acceptance)

Purpose: catch a **shared** misreading between the QPN2 spec and the repack by
comparing model outputs, not weights. A byte cross-check against 1Cat would
catch a difference; it cannot catch both implementations agreeing on a wrong
lane/K order. The current Marlin path is an independent, known-good consumer of
the same checkpoint, so output equality between the two paths is the functional
check.

## 1. Single variable

Both arms run the **same commit**, the **same checkpoint**
(`Qwen3.8-Flash-Next-NVFP4`), the same 4-card island and the same launch flags
(TP4, EP1, page 64, context 8192, MTP 3/1/4, rejection sampling, mem fraction
0.82). Only the decode MoE layout differs:

```
SGLANG_SM70_FP4_MOE_LAYOUT=marlin|qpn2   (default marlin)
```

The QPN2 arm repacks the expert weights at load time into QPN2; the Marlin arm
repacks into Marlin. Nothing else changes.

## 2. What is switched

First switch **only the decode-shaped expert GEMMs** (`M <= 8`) of the MoE
layers: `w13` (gate_up) and `w2` (down). Everything else stays on its current
path in both arms, so it cannot confound:

- prefill expert GEMMs stay on Marlin until the P3 reader exists;
- attention, shared experts, router, norms, embeddings, LM head unchanged;
- the MTP draft head uses the same target MoE weights; record its accept length
  so a draft-only error is visible even when the target tokens match.

Localization toggles, all load-time:

```
SGLANG_SM70_QPN2_TENSORS=both|w13|w2
```

`w13` is N=320 / K=2560 (tiles=10, groups=160); `w2` is N=2560 / K=160
(tiles=80, groups=10) at TP4. These two exercise the tile split and the group
split respectively.

## 3. Prompt set

Fixed, greedy, batch size 1 (`--max-running-requests 1`), 128 output tokens.

| axis | values |
|---|---|
| context | 76, 993, 3,473, 7,168 tokens |
| content | (a) repeated technical seed from `scripts/bench.py`; (b) unique RNG word prompt, fixed seed per context |
| prompts | 2 per (context, content) |

That is 16 prompts. One warmup request per prompt (JIT/capture) is discarded;
then 2 timed reps per prompt. The 7,168-token cell is the long-context/address
case; 76 is the shortest decode; 993/3,473 are the recipe rows.

Record per prompt: the full 128 greedy token IDs, the per-token logprob of the
chosen token (and top-2 logprobs at each position), the accept length and
verify count. Run one **MTP-off** arm at 993 tokens to separate the target from
the draft.

## 4. Tolerance and pass/fail

Both arms consume the same weights, so the only expected difference is
accumulation order (different layouts/kernels). A layout error permutes K or
columns and changes the expert output by O(1); it is not a small numeric
difference.

- **Pass**: all 16 prompts x 2 reps have **identical 128-token greedy
  sequences**, and the max per-token `|delta logprob| <= 1e-3` nats.
- **Near-tie pass**: a token mismatch is accepted only when, at that position,
  the Marlin top-2 logprob gap is `< 1e-3` nats and the QPN2 token is the
  Marlin second choice. No mismatch is allowed in the first 8 generated
  tokens.
- **Fail**: any mismatch with `|delta logprob| > 1e-3`; any systematic shift,
  repetition or truncation; any crash; any `w13`-only or `w2`-only arm failing
  while `both` passes is also a fail (it means the combined path masks a
  per-tensor bug).
- **Draft check**: accept length in the MTP arms must stay within 0.05 of
  Marlin. A larger drop with identical target tokens localizes to the draft
  path.
- A single-layer kernel check runs first (below); the model guard is accepted
  only if that passes.

## 5. What a failure localizes

| observation | localization |
|---|---|
| `w13`-only fails, `w2`-only passes | gate_up layout: tile/lane mapping (N=320, tiles=10) |
| `w2`-only fails, `w13` passes | down layout: group split/stride (K=160, groups=10) |
| both fail at token 1 | expert output from the first decode call; check the w13/w2 addressing and the expert stride |
| tokens match, logprobs differ by a constant scale | scale path: fp16 RNE vs float32, or the global-scale multiply |
| divergence only at 7,168 tokens | address overflow / padded or last-tile handling |
| divergence grows with position | accumulation or a per-layer stride error; bisect layer index if a layer toggle is added |
| MTP accept drops, target tokens match | draft path uses the same QPN2 expert GEMM and is wrong there first |

Record the first divergent prompt, position, the two top-2 logprob vectors and
the token IDs. That is the artifact 9b needs.

## 6. Single-layer check before the server

One full-size MoE layer at `M = 1..4`, same input on both paths:

- gate_up: N=320, K=2560; down: N=2560, K=160 (TP4 per-rank shapes);
- compare the QPN2 GEMV output with the Marlin output: max relative error
  `<= 1e-3`, and the same top-1 expert-route contributions;
- this runs without the server and localizes a kernel bug before the 4-minute
  model load.

## 7. Cost and order

1. single-layer check (seconds, one card);
2. Marlin vs QPN2 `both` two-launch guard (~4 min load each + ~2 min requests);
3. only on failure, `w13`-only and `w2`-only launches (two more loads).

Acceptance evidence: the token/logprob table, the accept lengths, and the
first-divergence artifact, recorded in the lab note for P2.
