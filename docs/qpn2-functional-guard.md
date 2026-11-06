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

The bank itself is `scripts/qpn2_probe/fixture_qpn2.py`; both arms load it
instead of building their own inputs:

```sh
python scripts/qpn2_probe/fixture_qpn2.py build --out DIR   # CPU, ~10 s, 11 MB
python scripts/qpn2_probe/fixture_qpn2.py verify DIR
```

It carries, per shape, the ModelOpt checkpoint tensors (`codes` [E, N, K/2],
`scale_bytes` [E, N, K/16] E4M3, raw `global_scale` [E]), the QPN2 bytes from
the independent oracle, one activation row per route, the shipping routing
metadata (`sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`, block 8,
top-k 1) and a second case carrying token 0's two top-k routes (top_k 2, route
indices 0 and 1), each writing its own output row, so the multi-route path is
exercised; the 2x sum belongs after the top-k reduce, not in the row. The dense
reference and expected outputs are derived on load from the codes and scales,
so a packed-buffer bug cannot hide behind a reference built from it.

Every input is a prefix of a SHAKE-256 stream, not a torch RNG draw: torch does
not promise a stable stream across versions. `verify` rebuilds from the streams
and compares SHA-256s; the manifest is byte-identical across hosts (checked on
two numpy versions). The fixture's QPN2 bytes equal 9b's `pack_qpn2` on both
shapes (8/8 experts, `max|delta| = 0`).

The Marlin-layout bytes come from `build --with-marlin`, which runs the shipping
`_repack_experts` / `encode_scales` / `encode_global_scale` once on a GPU host
and stores the result. There is no CPU reimplementation of
`gptq_marlin_repack`: the Marlin arm is the control, and a second implementation
would be a second thing that can be wrong. Without that flag the harness calls
the shipping conversion itself.

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

## 4a. Running the comparator

`scripts/qpn2_probe/compare_runs.py` implements section 4 and the section 5
table, so the P3/P4 run is turnkey:

```sh
python scripts/qpn2_probe/compare_runs.py runA.json both=runB.json \
    [w13=runB13.json w2=runB2.json] [--json verdict.json]
python scripts/qpn2_probe/compare_runs.py --selftest
```

Each run writes one JSON artifact (schema in the script docstring): launch
metadata (`layout`, `qpn2_tensors`, `mtp`, `commit`, `tp`, `model`,
`checkpoint`), and per prompt the 128 greedy token ids, the chosen token's
logprob at each position, the top-2 logprobs and token ids at each position
(the near-tie rule reads the baseline's), and the accept length. The comparator
prints the per-prompt table, the cause from the section 5 table, and a verdict;
it exits 0 on pass, 1 on fail, and 2 when the inputs cannot be judged
(incomparable launches, MTP on, malformed artifact).

It refuses MTP-on artifacts unless `--mtp-secondary` is given, because rejection
sampling makes token equality unattributable; with that flag it judges accept
length and logprob drift only, per the MTP-on note above.

`--selftest` runs 17 synthetic checks, one per failure class plus the near-tie
pass, the near-tie-inside-the-prefix failure, the per-tensor arm mapping and the
MTP path. Run it after any change to the comparator and before it judges a real
run.

## 4b. Collecting the artifacts

`scripts/qpn2_probe/run_artifact.py` drives a running server and writes one
`qpn2-run-v1` file per arm:

```sh
python scripts/qpn2_probe/run_artifact.py collect \
    --base-url http://lambda4:30000 --out runA.json \
    --layout marlin --commit "$(git rev-parse HEAD)" \
    --model Qwen3.8-Flash-Next --checkpoint "$CKPT" --tp 4 \
    --tokenizer "$TOKENIZER"
# then the same against the QPN2 launch:
#   --layout qpn2 --qpn2-tensors both --out runB.json
python scripts/qpn2_probe/compare_runs.py runA.json both=runB.json
```

Each request asks `/generate` for `return_logprob`, `top_logprobs_num=2`, greedy
sampling and `ignore_eos`, so every prompt returns exactly `--max-new-tokens`
(default 128) tokens. The artifact records the launch metadata the comparator
checks (`layout`, `qpn2_tensors`, `mtp`, `commit`, `tp`, `model`, `checkpoint`),
and per prompt the ids, the chosen-token logprobs, the top-2, the accept length
(`spec_accept_length`) and `verify_ct`. The 16 prompts are the section 3 set,
and they are identical across arms because they are built from fixed seeds, not
from a per-request salt.

The harness is testable without the engine:

```sh
python scripts/qpn2_probe/run_artifact.py selftest       # fake /generate + both arms + the comparator
python scripts/qpn2_probe/run_artifact.py fake-server --port 18400 --mode diverge
python scripts/qpn2_probe/run_artifact.py collect --dry-run --out a.json --layout marlin --commit dry
```

The fake server serves well-formed canned responses in three modes (`match`,
`diverge`, `offset`). The selftest asserts the request carries the logprob flags,
that the artifact is schema-shaped, that matching arms pass the comparator end to
end, that an offset arm fails as `scale-path`, a diverging arm as `layout`, an
unreachable server as `crash`, and that `--dry-run` writes a 16-prompt artifact
with no server.

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
