# Speculative decoding schemes: what is obtainable on SM70

Item 3 of epic #76. The verify machinery (one target forward over `1 + draft`
positions, accept the longest correct prefix, roll back KV) is shared by every
scheme; what differs is where the draft comes from. This page records, per
scheme, what a launch can actually use today and what it costs.

## The matrix

| scheme | draft source | usable today on | draft size | measured on SM70 | slot on a 32 GB card |
|---|---|---|---|---|---|
| MTP / NEXTN | carried by the checkpoint | Qwen3.8-27B-FP8, Flash-Next NVFP4 | 0.44 GiB (27B, 22 tensors), 4.86 GiB (Flash-Next, 31 tensors, includes its expert set) | 27B: accept length 5.82 of 6 draft tokens; Flash-Next: 3.77 of 4 | yes on both; on the 27B at TP2 it is 0.44 GiB per rank pair |
| DFlash | separately trained draft | Qwen3.8-27B (`Qwen3.8-27B-DFlash2-fp16`) | not measured | accept rate 0.22-0.47; 62.2 / 62.4 / 49.9 tok/s at 1.7K / 6.7K / 27.4K, 14-17% under the MTP head on the same bench | yes, but it loses to the head the checkpoint already ships |
| EAGLE-2 / EAGLE-3 | separately trained draft | none | - | - | not testable until a matching draft checkpoint is published |
| DSpark | separately trained draft | none | - | - | not testable until a matching draft checkpoint is published |
| n-gram / prompt lookup | training-free | any model | 0 | not measured on SM70 | free, but the accept rate only pays on repetitive prompts (RAG, code completion) |

Sizes are sums of the `mtp.*` tensors in each checkpoint index, at the
precision the checkpoint stores. The 27B stores FP8, so its in-memory head is
about twice that.

## What the numbers say

**MTP is the only scheme with a real choice to make.** It ships with the
checkpoint, so there is no separate artifact to obtain. On the 27B it is worth
keeping below 64k and turning off above it: the draft decode grows with the
prefix and overtakes the accept gain between 32k and 64k
(`mtp-crossover-2026-11-04.md`, and the recipe's measured table).

**A separately trained draft has to beat the one in the checkpoint.** DFlash2
is the only one obtainable for these models, and on the 27B it lands 14-17%
under MTP at every prompt length measured. That is the bar any other scheme
must clear, and it is why none of EAGLE or DSpark are on the work list: without
a published draft there is nothing to measure.

**The n-gram path is the only one that needs no artifact.** It has never been
measured on SM70, and its accept rate is a property of the workload rather than
of the model, so a single number would not generalize.

## What is shared, and therefore what a new scheme costs

Every scheme reuses the target's verify pass and the draft-decode container.
For MTP/NEXTN the draft decode is the Triton multi-step backend
(`TritonMultiStepDraftBackend`). At 130k on the 27B it costs 80.3 ms per step at
the default 8 KV splits and 38.2 ms at 80 splits, against 1.3 ms for one layer
of the native verify kernel over the same KV (`decode-at-130k-splits.md`). That
cost is shared by every scheme that drafts autoregressively, which is why #76
item 2 is ahead of any new scheme in this list.

## Not confirmed

- Whether an EAGLE-3 draft exists for Qwen3.8-27B or Flash-Next. Confirmation
  method: search the model publisher's release page for a draft checkpoint.
- The n-gram accept rate on SM70. Confirmation method: launch with
  `--speculative-algorithm NGRAM` at 32k on a repetitive prompt and read
  `spec_accept_length` from the `spec-accept` row the GPU suite now records.
