# Status: where the SM70 work stands

As of **2026-11-18**. `main` is at `3e5d61708d`; the SM70 branch is at
`d9d6303939`. Nothing in this page is a serving result unless it says so, and
every number carries the arm it came from.

## One line

Bluewhale (DeepSeek-V4.1-Flash-1017, base-fp8) **loads end to end on V100** with
real fp8 dense and real MXFP4 experts; the **attention component** is the last
thing between it and a token, and stage 1's torch decode path is written but has
never run against a model.

## What is on `main`

Patches 0001-0008, each one an upstream edit this hardware needs:

| Patch | What it fixes | Evidence |
|---|---|---|
| 0001 | HiCache OpenSSL prefix | pre-existing |
| 0002 | IPC tuple get | pre-existing |
| 0003 | QPN2 Marlin MoE shape | pre-existing |
| 0004 | DeepSeek-V4 model class unimportable without DeepGEMM | import A/B under `--nv`; `EntryClass` registers |
| 0005 | Four SM70 env defaults (`DEEPGEMM_HC_PRENORM`, `FP8_PAGED_MQA_LOGITS_TORCH`, `TILELANG_MHC_PRE/POST`) | applied, imports; no served-model run |
| 0006 | MHC prenorm `hidden_block` selected by rule, not a size table | **inert on SM70** — the sm70 MMA macro generator has no fp32 load layout |
| 0007 | DSV4 base constructible without `flash_mla` (named refusal at the terminal call) | mechanism test; no constructed backend yet |
| 0008 | DSV4 KV store fp16->bf16 cast at `store.cuh:76` | compiled on V100, unpatched rc=1 / patched rc=0, bit-exact read-back |

Also on `main`: `scripts/pack_evidence.sh` (fail-closed evidence packing) and an
exclusion-based receipt trigger, so any file under `sglang_restomod/`, `patches/`
or `recipes/` needs a receipt — only `.md`, `tests/` and `cli.py` are exempt.

## What is on the branch, not on `main`

`sglang_restomod/kernels/` — three gated components, none A/B'd in an engine
because no DeepSeek-V4 forward exists yet:

| Component | Switch | Wired behaviour | Status |
|---|---|---|---|
| `deep_gemm_sm70` | `SGLANG_SM70_DEEPGEMM_SUBSTITUTE=1` + patch 0009 | `sys.modules["deep_gemm"]` substitute, SM70 only, never shadows a real install | three GEMM entries are **permanent refusals** (forwarding would duplicate quantisation); opt-in, default off |
| `hc_prenorm_sm70` | `SGLANG_SM70_HC_PRENORM=torch\|fused\|auto` | `auto` = torch below M=1024, kernel at or above | 1.32x at 4096, 1.41x at 8192; **loses 0.84x at 256** — prefill only |
| `paged_logits_sm70` | `SGLANG_SM70_PAGED_LOGITS=torch\|fused\|auto` | `auto` = kernel at B<=4, torch above | 2.25x at B=4 (~1.2 ms/step); 1.94x at B=32 **misses its 3x gate**; TileLang needed for the large-batch prize (~5.2 ms/step) |

Plus `sglang_restomod/attention/dsv4_sm70/` — attention stage 1: `kv_layout.py`
plus `torch_attn.paged_mla`, with 13 CPU tests. Sparse prefill raises by name.
**This has not run against a model.**

## Measurements that changed decisions

These are the numbers that overturned an earlier reading; each is recorded with
the retracted claim beside it, because the retraction is the useful part.

| Measurement | Result | What it overturned |
|---|---|---|
| `hc_pre_torch_impl` eager vs CUDA graph | 0.114 -> **0.027 ms** (M=1) | "prenorm costs 4.7 ms/step in decode" — in-graph it is 1.06-1.85. A fused kernel for the same call only went 0.223 -> 0.189, so it was never launch-bound |
| indexer torch, captured | B=1 0.140, B=8 0.433, B=32 1.343 ms | "flat from 2048 to 8192 pages" — it was one batch over a narrow range |
| Triton dtype sweep, B=32 | fp32 0.86, fp16 0.87, **bf16 9.2-9.9 ms** | "use 16-bit MMA" — under Triton on sm70 fp32 is the fast path; only TileLang reaches an MMA |
| MHC prenorm codegen | `mma_sm70_macro_generator.py:359` `ValueError: Unsupported dtype float32` | "the size whitelist blocks 20480" — the constraint was dtype, not size |
| TP2 Flash-Next | OOM at `31.34 of 31.73 GiB` on both ranks | "TP2 is enough" — a property of the model on 32 GB cards, not of the host |

## Bluewhale: what works, what does not

Verified:

- a real 2.4 GB slice (`models/bluewhale/mini-real`, layers 2-3 of base-fp8
  renumbered to 0-1, real fp8 dense, real MXFP4 nibbles and E8M0 scales) loads
  end to end: `Load weight end. elapsed=2.85 s, mem usage=3.84 GB`;
- both expert layers report `prepared for marlin_v100 (E8M0 per 32 -> E4M3 per
  16)` **on real checkpoint scales**, no clamp error, no missing-global check;
- an MXFP4 expert GEMM executes on SM70 (96 calls, 2 distinct experts per call)
  on the dummy arm, and the global-scale fix is exercised on real scales.

Not working yet:

- **no forward has run.** `hc_pre_torch_impl` and `post_torch_impl` have never
  executed on this hardware;
- attention is the wall: every `--dsv4-attn-backend` choice inherits a base that
  imports `flash_mla` (Hopper-only), so the component replaces
  `flash_mla_with_kvcache` and `_forward_prefill_sparse` — two terminal calls,
  with the metadata layer reused and the indexer already torch on SM70;
- **the store fix (0008) covers the 584-byte V4 layout; mini-real is the
  528-byte V4.1 layout**, so which layout the write path takes for our artifact
  is the next thing to establish, not something already closed;
- capacity: base-fp8 is **38.5% expert-resident on 4 cards**. Even with attention
  working, a serving bluewhale needs an expert-offload story that does not exist.

## Next, in order

1. **Attention stage 1 -> a token.** Torch decode against `mini-real`, then torch
   prefill with `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0` forced so the dense
   fallback is reachable. Acceptance is a token with both checks, not a
   half-written backend.
2. **Establish the KV layout for V4.1** (528 bytes) and whether 0008's cast is
   on that path.
3. **A/B both kernels in that same serve** — prenorm at prefill shapes, paged
   logits at the served batch — one window, two answers, one baseline.
4. **TileLang rewrite of the paged-logits kernel** if the B=32 prize (5.2 ms/step)
   is wanted; Triton cannot reach an MMA on this part.
5. **Expert residency.** 38.5% on four cards is the binding constraint on
   serving, and it is a capacity problem, not a kernel.

## Open decisions

- `strict_required_status_checks_policy` is `true` on `main-protection`, so a
  receipt-bearing PR that falls `BEHIND` needs a fresh signature before merge.
  Recommendation: turn it off; one signature then lasts until merge. Not changed.
- Three published evidence bundles carry the operator account name. They were
  **not** rewritten (a receipt cites a digest), and their release notes disclose
  it. Republishing means three fresh receipts; recommendation: leave them.

## How to pick this up

```bash
python scripts/check_pin.py                  # pin OK: v0.5.23 (297e54f...)
python scripts/check_patches.py --skip-apply # 8 patches
```

Read, in this order: `docs/sm70-dsv4-attention-plan.md`,
`docs/sm70-indexer-kernel-design.md`,
`docs/runs/2026-11-17-fallback-microbench.md`,
`docs/runs/2026-11-17-indexer-graph-capture.md`,
`docs/runs/2026-11-17-prenorm-kernel-gate.md`.

Three habits this work paid for, all recorded in `docs/gpu-receipts.md` and the
component PROVENANCE files: **measure in the regime that runs** (captured, small
batch); **an instrument that cannot fail proves nothing** (no-GPU import A/B,
header-only TU); **state the diff range with the claim**, because every session
here commits as the same author.
