# 2026-11-16 — How both references run bluewhale on 8 cards or fewer

Status: **open, source-read**. One table for the three-tier budget, one sentence per
system on the mechanism that actually buys the capacity, and what it means for our
stack. Not part of the decode study.

Sources: Strata's own planner `strata/strata_control/dsv41_fit.py` (read by the
coordinator, constants quoted below); 1Cat-vLLM at
`1CatAI/1Cat-vLLM@ad3b337` (read directly, file and line cited); our checkpoint
byte map (§6 of the capacity page).

## 1. The answer in one line each

- **Strata**: they do not fit it in VRAM, and they do not try to. Experts beyond
  the VRAM tier are **computed on the CPU**, reading model files in place through
  the page cache. The Engram table never enters memory.
- **1Cat-vLLM**: the cold tier is **copied, never computed** — a pinned host arena
  holds non-resident experts and a GPU kernel loads them into VRAM slots before the
  multiply, with "no full FP16 expert library or CPU matrix fallback"
  (`benchmarks/deepseek_v41/README.md:25`). Only the Engram is read from disk.
- **Ours**: one tier. Experts must be **fully resident**, because
  `--cpu-offload-gb` is mutually exclusive with PLE/Engram offload
  (`srt/arg_groups/memory_hook.py:30-36`). We have the Engram tier and no expert
  tier.

**So the two references answer different questions.** Strata buys capacity by moving
*compute* off the card; 1Cat buys it by moving *bytes* into the card on demand and
by choosing a set small enough to fit. Ours moves neither.

## 2. The budget table

Per-card expert capacity uses one formula for all three columns so the rows are
comparable: `n x (34.36 - dense_per_card - 0.62 - 3.6)` with
`dense_per_card = (5.62 + 0.24n)/n`, all from Strata's planner.

| | Strata | 1Cat-vLLM | ours |
|---|---|---|---|
| expert set | 2.25 bpw GGUF, **149.06 GB** | GSQ-RCO **IQ2_XXS** (types 16/17/29) — README states 132.07 GB routed | base **FP4, 288.76 GB**; or the 2.25 bpw set, 149.06 GB |
| VRAM expert capacity | 114.0 GB at 4 cards | 26.36 GB/card at 4 cards, 105.4 GB | 114.0 GB at 4, 233.6 GB at 8 |
| resident fraction | **74.6%** at 4 cards (projected); **38.5%** measured, because the measured run used the FP4 set | 4-card profile keeps a 30% cold set (39.55 GB shared, 26.36 GB/card GPU) | **100% required** |
| cold tier | **CPU computes** from the model files, page cache | **pinned host RAM**, DMA into GPU slots by kernel — no CPU matmul | **none available** |
| Engram tier | SSD, rows per token | **disk**: node-level `pread` into a shared memfd arena, 24 rows/token/channel, deduped, 64 reader threads (`csrc/gguf_shared_engram.h:79-88`, `engram_io.py:403-419`) | host-pinned (PLE), 47.75 GiB on the served model |
| cards for the whole model | 4 to run, 8 to hold everything | **8 = PP2 x TP4, every expert resident** (`serve-pp2.yaml:4`, cold set zero) | **6** at 2.25 bpw, **10** at base FP4 |

Projection caveat, stated once: the 74.6% and 38.5% rows are the same planner
budget applied to two different artifacts. The planner's *measured* lines
(6,197 experts = 116.5 GB, 9,163 on the CPU lane = 172.3 GB) sum to **288.8 GB**,
the FP4 expert total exactly, so the run that produced them was on the **FP4** set
at 18.80 MB/expert — not the 2.25 bpw set at 9.70 MB/expert. Mixing the two puts
the projection a factor of **1.94** out.

## 3. What each does that the other does not

**Strata computes on the CPU; 1Cat moves bytes.** That single difference decides
the host requirement: Strata needs CPU throughput and a host page cache, 1Cat needs
**pinned host RAM and PCIe bandwidth** but no second compute engine. 1Cat's README
rules out the CPU lane in as many words, so their cold tier is a latency and
bandwidth problem, not a compute one.

**1Cat's host tier is one pool per TP group, not per rank** — "one size-sealed
memfd per TP group, mapped with `MAP_SHARED`" (`README.md:52-55`). Shared cold
storage means four ranks do not each hold a copy, which is the duplication we would
otherwise pay at TP4.

**1Cat's Engram tier is engineered for exactly the access pattern measured here** —
24 rows per token per channel, deduplicated inside the request, read by one
node-level reader, "never replicating the complete tables" (`README.md:35-36`).
That is an independent confirmation of §6.4's ~530 B/token figure: at 264 bytes per
row, 24 rows per channel is a *burst* ceiling, not a per-token average.

**Neither uses stock `--cpu-offload-gb`.** 1Cat's mechanism is
`sm70_gguf.expert_cache_host_storage: shared_original`
(`vllm/config/kernel.py:568-575`) with a native cache coordinator
(`csrc/gguf_expert_cache.cpp`); stock UVA offload still exists in the tree but is
not referenced by the GGUF/V4.1 path. So the answer to "is it just stock offload"
is no — and that matters, because stock offload swaps whole layers and cannot make
a per-expert residency decision.

## 4. What it means for us

**Eight cards is reachable for us only with the 2.25 bpw set.** At base FP4 the
experts are 288.76 GB against 233.6 GB of capacity at 8 cards — short by 55.2 GB —
so the published checkpoint needs **10 cards** fully resident. At 2.25 bpw it is
149.06 GB and fits **6**.

**We cannot buy the difference the way either reference does.** Strata's answer
needs a CPU expert lane, which we do not have and which 1Cat explicitly declines to
build. 1Cat's answer needs a pinned host arena with in-kernel loads, which is a
mechanism our engine does not have for experts at all — and the one generic knob
that would approximate it is the one disabled by PLE offload. **Our card count is
therefore set by full residency, and the only lever we hold is the quantization of
the expert set.**

**PP2 x TP4 is a shape we do not have.** 1Cat's 8-card recipe is pipeline-parallel
2 by TP4; our serving profile is TP4 on one host, and cross-host TP4 was measured
at 66% of prefill in all-reduce. So their 8-card arrangement is not directly
transferable even if the byte budget is.

## 5. Not verified

- Strata's measured numbers are **relayed** from its planner docstring as read by
  the coordinator, not re-read by me; the source is not on this host.
- 1Cat's README figures (132.07 GB routed, 26.36 GB/card, 39.55 GB cold) are
  **their** measurements on **their** hardware; nothing here re-derives them.
- Whether 1Cat's Engram reader sustains the 8-card profile at 1M context is
  **unmeasured** — the 24-rows-per-channel figure is a design bound, not a
  throughput result.
- Our 6- and 10-card figures are **budget arithmetic** on measured byte counts, not
  a launch. No bluewhale serve of ours exists.
