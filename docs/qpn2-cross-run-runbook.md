# QPN2 cross-run runbook (P3/P4)

Operational companion to `docs/qpn2-functional-guard.md` (design and
tolerance) and `recipes/flash-next-nvfp4-v100-tp4-mtp.md` (flags). This file
is the checklist and the exact commands for the two-run comparison. **Do not
start it** until every precondition below is met: QPN2 cannot serve a prefill
shape today, and the component raises rather than falling back.

## 0. Why two runs, not one process

Both layouts are 0.5625 B/weight. At TP4 the production checkpoint is 29.6-30.0
GB per card in one layout, so holding Marlin and QPN2 at once does not fit in
32 GB. Each run holds exactly one layout, and the comparison is across runs on
identical inputs. The P2 fixture guard (one process, both layouts) covers the
decode kernel; this run covers the model.

## 1. Preconditions

- [ ] df's QPN2 prefill reader is merged on the QPN2 branch and its own guard
      passes on `M > 1` (spec §7). Without it, run B cannot load a model.
- [ ] `feat/qpn2-layout-224` contains no change other than the QPN2 work
      relative to its base (`f8207e13e9`). `git diff --stat` between the two
      commits is the review surface.
- [ ] The P2 fixture guard passes at the branch head (worst 6.4e-4 on
      2026-11-06).
- [ ] Four cards on one NVLink island free. `--ep-size 1`.
- [ ] Same checkpoint path and same `UPSTREAM_PIN.toml` in both runs; record
      the checkpoint manifest/mtime and the `.so` hashes (`sha256sum
      $SM70_MARLIN_DIR/*.so`) in both run logs.

## 2. Runs

Run A (control, Marlin everywhere):

```sh
cd <worktree at f8207e13e9>
SGLANG_RESTOMOD_COMPONENTS=attention_v100,fp4_moe,moe_align_sm70,ple_sm70 \
SGLANG_PLUGINS=restomod SM70_MARLIN_DIR=<dir> \
  python -m sglang.launch_server \
    --model <Qwen3.8-Flash-Next-NVFP4> --dtype float16 --quantization modelopt_fp4 \
    --moe-runner-backend marlin --tp-size 4 --ep-size 1 \
    --attention-backend qsa_v100 \
    --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
    --sampling-backend pytorch --ple-offload-embedding \
    --context-length 4096 --page-size 64 --mem-fraction-static 0.82 \
    --max-running-requests 1 --chunked-prefill-size 4096 \
    --cuda-graph-backend-prefill=disabled
```

Run B (QPN2 everywhere):

```sh
cd <worktree at feat/qpn2-layout-224 head>
SGLANG_RESTOMOD_COMPONENTS=attention_v100,fp4_moe,moe_align_sm70,ple_sm70 \
SGLANG_PLUGINS=restomod SM70_MARLIN_DIR=<dir> \
SGLANG_SM70_FP4_MOE_LAYOUT=qpn2 SGLANG_SM70_QPN2_TENSORS=both \
  python -m sglang.launch_server ...   # identical flags to run A
```

**MTP is off in the primary arm** (no `--speculative-*` flags): rejection
sampling is stochastic even at temperature 0 (#217), so token equality across
runs cannot be attributed to the layout with it on. A secondary MTP-on run
records accept length and logprobs only.

If run B raises `QPN2 expert weights reached a shape the QPN2 decode kernel
does not cover`, the prefill reader is not on the path: stop, that is a run B
configuration failure, not a layout result.

## 3. Prompt set and sampling

Per guard doc §3: fixed greedy, batch 1, 128 output tokens; contexts 76, 993,
3,473, 7,168; two contents (technical seed, RNG word prompt with a fixed seed)
x 2 prompts. One warmup request per prompt discarded, then 2 reps.

## 4. Artifacts

Per prompt, per rep: the 128 token IDs, the logprob of the chosen token at each
position, the top-2 logprobs, and (MTP-on only) accept length and verify count.
Store one JSON per run plus a diff report; the lab note for P3/P4 keeps the
table and the first-divergence artifact.

## 5. Pass/fail

Guard doc §4:

- pass: identical 128-token greedy sequences on all prompts and reps, and
  `max |delta logprob| <= 1e-3` nats;
- near-tie pass: a mismatch is accepted only when the control's top-2 gap at
  that position is `< 1e-3` nats and the QPN2 token is the second choice; no
  mismatch is allowed in the first 8 generated tokens;
- fail: any other mismatch, a systematic shift/repetition/truncation, a crash,
  or a per-tensor arm failing while `both` passes.

## 6. If it fails

Use the localization table in guard doc §5 (`w13` vs `w2`, first token vs long
context, tokens vs logprobs). The fixture already localizes the decode kernel;
a failure here that the fixture does not reproduce points at prefill or at the
address/stride over the whole model.

## 7. Status this unblocks

The P1 probe result (gate_up 35.0 us / 527 GB/s, 4.7e-4) stays **provisional**
until this run passes. The decode-only P2 fixture guard does not stand in for
it: it proves the kernel, not the served model.
