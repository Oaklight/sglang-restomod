# Flash-Next on V100: support status and inference performance

Status: **partially verified.** Each claim below carries one of three states:

- **已验证** — a run in `docs/runs/` says so, and the page names its arm.
- **未验证** — no run behind the claim here, or the run is reported without a
  trace this repository holds.
- **未确认** — a page states it, but the measurement or definition is open.

Numbers are one per row, with the arm and the date. Nothing is aggregated, and
nothing is quoted that its record marks retracted.

## What runs

One host, four V100-SXM2-32GB (SM70), TP4 with one card per rank. Model:
Qwen3.8-Flash-Next-NVFP4, our re-export with `mtp.*` left unquantized. Engine:
SGLang v0.5.23 plus this repository's three patches. **已验证** for "serves":
a served completion exists on this arm, and the per-card and decode pages were
taken on it.

Nine components must be selected:

`attention_v100`, `fp4_moe`, `ple_sm70`, `spec_sm70`, `moe_align_sm70`,
`gdn_tilelang_sm70`, `fp8_dense`, `fp8_lm_head`, `fp8_dense_quant`.

**An SM70 launch must name at least one component.** `SGLANG_PLUGINS=restomod`
alone installs the plugin and changes nothing, including the unmodified sm75
gate, so the launch refuses to start. Naming no component is a supported
baseline, not a misconfiguration: an unset or empty
`SGLANG_RESTOMOD_COMPONENTS` loads the plugin and does nothing on purpose, so a
bare plugin load cannot silently patch the engine. The list above is the
minimum set this model has been served with. Keep three facts apart: the plugin
is installed, a component is selected, and a component is active. Logs that
report the first two say nothing about the third.

**Selected is not loaded, and this repository has been burned by the gap more
than once.** The pages record a component that reported loaded while binding
nothing (`spec_sm70` before #309), a dispatch gate that admitted zero layers
while every log line looked healthy (the exact-dense prefill gate before #299),
and a kernel that never ran at all (QPN8). The census attestation that would
settle the current arm — nine selected, nine loaded, `failed=[]`, `missing=[]`,
`inactive=[]` — **is not a page in `docs/runs/`**. Treat "all nine loaded" as
**未验证** until an attestation is committed. What a reader can check today is
the component list in `/proc/<pid>/environ` and the admitted/active counts, which
the study protocol requires.

## What is inert, conditional, or was

| item | state | evidence |
|---|---|---|
| `spec_sm70` | **was inert; fixed in #309.** `register()` dropped `install()`'s result, and it replaced flashinfer attributes rather than the `LazyImport` handles the pin actually calls. Fixed by replacing the handles. **已验证** for the fix (the PR carries GPU runs and its receipt check passed). | #309 |
| exact-dense prefill gate | **admitted zero layers until #299.** The guard read `layer.tp_size`, which no linear in the pin carries, so it evaluated `1 != 4` and refused every candidate while other gates reported success. Fixed. **已验证** for the fix. | #299 |
| QPN8 | **default 0 since #306.** It runs only with `SGLANG_SM70_FP8_DECODE_QPN8=1` and the Flash-Next shape table. "Neutral" is not established in any page; enabling it is a decision with a number to take. **未确认.** | #306 |
| TurboMind dense FP8 | **not on unless `SGLANG_SM70_TURBOMIND_OPS_PATH` is exported.** Without it `fp8_dense`, `fp8_lm_head` and `fp8_dense_quant` register and do nothing; dense runs bf16. Two configurations are both called "served": S1 without the variable, S2 with it (416 layer binds). Cite which one. **已验证.** | per-card overhead run |

## Performance

### Decode

| quantity | value | arm and date | state |
|---|---:|---|---|
| step | **24.76 ms** | TP4, four cards, SPEC=1; 2026-11-16 capacity model | 已验证 |
| marginal accept | **3.84** | same arm; the printed mean is 3.12 — different definitions, do not mix | 已验证 |
| slope | **155.1 tok/s** whole model, **38.8** per card | same arm | 已验证 |
| per accepted token | **6.45 ms** | same arm | 已验证 |
| dense cluster | **40.2%** of verify-window kernel time, **2.40 ms/step** | one partial verify trace; the capture holds about one step in three | 已验证 |
| collectives | **0.22 ms/step**, **3.7%** of verify kernel time, 8.1 us/call | same trace | 已验证 |
| GEMV path | **8.2% of the step**; disabling it costs +2.07 ms (25.29 → 27.36 ms), above the 3.8% drift floor | 2026-11-16 decode GEMM gap | 已验证 |
| dense cluster, bf16 arm | **11.27 ms of 28.56 ms (39.5%)**; after block FP8, 9.53 ms; the remaining 6.42 ms is 97 HC linears and is launch-bound | 2026-11-15 dense-GEMM dive | 已验证 |

**Do not use 126.2 tok/s or 31.6 per card.** Both were retracted on 2026-11-16:
they divided one run's mean acceptance length by a step time differenced across
two runs, and the rate is 23% low. The step is unaffected.

### Prefill

| quantity | value | arm and date | state |
|---|---:|---|---|
| served arm, GDN pair on, dense FP8 active | **3912 tok/s** (1053 ms at 4121 tokens, spread 1.5%, quiet box); **3641** (1132 ms, other work on the box) | SPEC=1, 2026-11-15 competitor numbers | 未验证 — reported, trace not held here |
| same tree and prompt, GDN pair off | **1476 tok/s** | `SM70_GDN_TILELANG=0`; 2026-11-15 attribution note | 未验证 — the A/B has not run on a clean island |
| GDN kernel pair, isolated | **108x** and **33x** on the two kernels | fixed-seed tensors, one free card; 2026-11-15 GDN record | 已验证 |
| GDN end-to-end effect | **1390 ms of kernel time saved**; **1.84x** patch-attributable; **1.96x** measured on a shared box with unequal arms | 2026-11-15 GDN record | 已验证 for the kernel saving; the multiple is a reference, not a target |

Quote the kernel saving, not the tok/s multiple. The 1.96x arm was contended and
the two sides used different `--mem-fraction-static` values; the record says so
itself.

### Per-card footprint

| term | value | state |
|---|---:|---|
| fixed, S1 (shipped launcher, TurboMind unset) | **5.11 GiB (5.49 GB)** | 已验证 |
| fixed, S2 (S1 + TurboMind path, 416 binds) | **5.20 GiB (5.59 GB)** | 已验证 |
| mamba cache, named term | **1.43 GB** (conv 0.04 + ssm 0.91 + intermediate 0.47 + conv 0.01) | 已验证 |
| residual after naming | 2.72 GiB (S1) / 2.77 GiB (S2) | 已验证 |
| CUDA context | 366 MiB, identical on four ranks | 已验证 |
| CUDA graph pool | 716 MiB | 已验证 |
| KV | **~0.93 KB/token** | 未确认 — one arm divided by its token count; the instrument prefers an M3 slope |

## What the numbers cannot be used for

1. **Token equality cannot gate a kernel swap.** The stack is not bitwise
   repeatable: final logits move 1.6e-2 nats cold-to-cold on identical input,
   and the pure-Triton control arm moves too. A token-level A/B measures the
   baseline's run-to-run variation. Use kernel-level numerics and a step-time A/B.
2. **A profiled run's `client_elapsed` is not the wall.** On this build it is
   inflated 7.2x by the profiler's stop-and-export. Quote unprofiled walls.
3. **Per-card and per-model rates are different metrics.** The 2.77x per-model
   figure in the capacity model is against a 2.25-bit reference model, not an
   engine comparison. Against a same-format reference it is level per output
   token on comparable arms (8.02 against 7.71 ms, within 4%) and 1.36x behind
   on the other host's slope (95.2 against 129.6). Level matched shards before
   comparing cards.
4. **The 24.7% collective share is not decode.** That figure is the
   whole-capture, prefill-weighted share. Inside verify spans the collectives
   are 3.7%.
5. **Retracted figures must stay retracted:** the 62% GDN prefill ceiling is a
   pre-fix number; the "draft MoE is 40% of kernel time" projection is withdrawn
   (#292); "the 8x grid does not fit Flash-Next experts" describes bluewhale,
   not this model; "TileLang GEMV is untested" is closed (#301); the profiled
   wall is not the real wall; 126.2 tok/s and 31.6 per card are retracted.

## Open

- The nine-selected/nine-loaded census attestation is not committed.
- The capacity model lists **4,001 tok/s** prefill with no arm or date; it is
  not reconciled with 3912/3641. Not used in the table above.
- QPN8's effect is unmeasured; only its default and gate are known.
- The TurboMind discovery (#286) prefill regression has no same-tree A/B, so its
  size is unconfirmed.
- The GDN pair's clean-island A/B and the component-level (`auto`/`whole`) ratio
  are both outstanding.
- Accept appears as 3.84 (marginal) and 3.12 (printed mean). Both are real; a
  document that quotes one must say which.

## Related

- [Per-card overhead: measured](runs/2026-11-16-per-card-overhead.md) and its
  [instrument](runs/2026-11-16-overhead-bench-spec.md).
- [The decode GEMM gap](runs/2026-11-16-decode-gemm-gap.md) and the
  [dense-GEMM dive](runs/2026-11-15-dense-gemm-dive.md).
- [GDN prefill kernels](runs/2026-11-15-gdn-tilelang-prefill-ab.md) and
  [attribution / TP4 placement](runs/2026-11-15-attribution-and-tp4-placement.md).
- [Capacity model](runs/2026-11-16-capacity-model.md) and
  [competitor numbers](runs/2026-11-15-competitor-numbers.md).
- The bluewhale engine page is in flight (PR #334); its run record is
  [bluewhale on eight cards](runs/2026-11-16-bluewhale-on-eight-cards.md).
