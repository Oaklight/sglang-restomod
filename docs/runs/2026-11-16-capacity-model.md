# 2026-11-16 — Capacity model for V100 serving

Status: **open**. Scope: capacity is a lever, not the goal. The goal is highest
inference performance on V100, so every term here is justified only by what it
does to throughput or to how many cards a load needs. Quality at a given bit
width is the quantization team's problem and is out of scope.

Target models: the served Flash-Next class, and **bluewhale** (the large-MoE
target; codename used throughout). Coefficients taken from a run follow
`docs/runs/2026-11-16-competitor-study-protocol.md`.

## 1. The two levers, and how capacity turns into throughput

- **L1 — model bytes per parameter** (quantization). Shrinks the whole set.
- **L2 — resident fraction per card** (placement). Decides how much of what L1
  produced must sit in HBM.

L1 and L2 together decide **how many cards a model needs**, and that is the only
route by which capacity reaches throughput here. Measured on the same model class,
competitor single card against our TP4:

| quantity | ours (TP4) | competitor (1 card) | ratio |
|---|---|---|---|
| expert bytes, **whole model** | 67.97 GiB | 33.1 GiB (cited 35.5 GB) | **2.06x** — L1 |
| expert bytes **resident per card** | 17.0 GiB | **26.50 GiB** (measured) | **1.58x in their favour** |
| residence as a **fraction of pairs** | 25% (÷4) | **80.2%** (measured) | **3.2x** |
| **cards to hold the whole model** | 4 | **2** (100% of pairs; 1 card holds 80.2%) | **2x** |
| prefill, whole model | **4,001 tok/s** | 1,108 tok/s | **3.6x in ours** |
| decode, **whole model** | **126.2 tok/s** | 55.9 tok/s | **2.26x in ours** |
| decode, **per card** | **31.6 tok/s** (126.2 ÷ 4) | **55.9 tok/s** | **1.77x in their favour** |
| decode, per **step** | 24.74 ms | 17.9 ms | 1.38x in their favour |
| decode, per **accepted token** | **7.93 ms** (accept 3.12) | 17.9 ms | **2.26x in ours** |

**Two metrics, opposite directions, and sharding is the reason.** They fit the
whole model on one card, so they pay no cross-card work per token; we pay 4-way
tensor parallelism, and the per-card figure is that bill — **though the census in
§5.1 puts collectives at only 0.199 ms per step, so the bill is not collective
latency. What it is remains open.** This is how capacity converts into throughput
— **not by reading fewer bytes, but by needing fewer cards.**

Measured per-card ratio: **1.77x** on the realised single-card run, **2.11x** using
the earlier 66.6 tok/s decode. The recorded **2.3–2.8x band sits above both** and is
reproduced by neither, so the page quotes the measured pair and retires the band.

**At matched shard count the per-card gap disappears.** Their two-card arm — 100%
of pairs resident, layers 0–23 on one card and 24–47 on the other — runs at
**62.9–70.7 tok/s total, 31.5–35.4 per card**, against our **31.6 per card on
four**. Per-card throughput is therefore set by **how many cards you shard
across**, not by the engine, and the single-card 1.77x is one card measured against
four. What survives as an engine difference is **capacity: 2 cards against our 4**
— and the second card buys only **+9–10% prefill and +12–21% decode** (56–63% of
single-card throughput per card), so it is capacity with a small latency gain, not
scaling.

So the capacity claim is narrow and strong: **two cards hold the whole model where
we need four**, and one card already holds 80.2% of it. Cross-config caveat, stated
once: their decode is 8192
context with int8 KV against our serving config, so no row here is a controlled
A/B.

## 2. Arbitration: two residency numbers that looked contradictory

The header constant said 4,105 slots = 5,413 MiB = **16.7%** of 24,576 experts.
The measurement says **19,954 slots = 26.8 GiB, 81.2%**, hit 98.5% on one card.

Both are real and they are **different settings, not different models**:

- 4,105 slots × 1,382,400 B = 5.68e9 B, i.e. it exactly fills a card with the
  header's stated **6,414 MiB free**. That is a small-headroom device profile.
- `--expert-cache auto` sizes to available memory. On the measured 32 GB card
  that is 19,954 slots. The header constant is a **floor, not the policy output**.
- 16.7% is also 4,096/24,576 — a power of two, i.e. a default.

**Model the measured value; keep the floor as a sensitivity point.** The same
distinction re-prices the host tier by an order of magnitude (§4), so it is not
cosmetic.

**A second discrepancy, also arithmetic, not error.** Expert bytes appear as
63.28 in one map and 67.97 GiB in another, and bits/param as 4.5 and 4.83. The
difference is exactly the MTP class: 63.28 + 4.69 (MTP expert tensors) = 67.97,
and 4.5 × 67.97/63.28 = 4.83. Routed experts alone are 4.5 bit/param; including
MTP they are 4.83. Both figures are correct; the class boundary must be stated
or the two tables will never reconcile.

### 2.1 Realised per-card resident set, competitor, single card

Three steady-state samples 8 s and 60 s apart are **identical on every byte
figure** — resident and stable, not a transient.

| class | realised |
|---|---|
| VRAM used | 31,923 MiB of 32,768 (320 MiB is a co-tenant's idle allocation) |
| **expert cache** | **19,721 slots, 26.50 GiB = 80.2% of 24,576 (layer, expert) pairs** |
| prompt-path borrow | 1,800 slots, 2.43 GiB, taken from the same cache |
| weights from pack | 1,466 MiB canonical + 300 native projection matrices at 1,474.85 MiB + 322 MiB Q5_K head |
| captured windows | 591 / 613 / 633 MiB |
| free with everything loaded | 729 MiB |
| host RSS / pinned arena | 62.1 GiB / 26.8 GiB |
| conversation cache | **70.3 KB/token** at int8 (282 MiB for 4,211 tokens) |

Slot arithmetic: 26.50 GiB / 19,721 = **1.442 MB** per slot against the header's
1,382,400 B — 4.3% higher for alignment and per-slot metadata. Use the realised
1.442 MB for capacity and the header figure for format arithmetic; the two differ
by design, not by error.

Policy, with absence recorded rather than inferred: the cache is **pre-filled from
`expert-profile.bin`** (19,721 slots in 2.6 s at 11,091 MB/s), and the R4 hit path
is on, with resident experts computed on the GPU. **Prefetch depth and eviction
order are printed nowhere in the run**; the flags exist (`--adapt-async`,
`--expert-cache-per-layer`, `--expert-cache-device1..3`) and sit at defaults. That
row stays "not printed", not "default".

## 3. L1 — bytes, and what can actually run

Per expert parameter, including scales: NVFP4 **4.5** (4 + 8/16); block-128 fp8
**8.25** (8 + 32/128); fp16 **16**. Their expert format is **2.25** (1,382,400 B
per 4,915,200-parameter expert), which reproduces their 35.5 GB pack from the
format alone.

| format | bit/param | 120.8 G expert params | SM70 prefill+decode kernel |
|---|---:|---:|---|
| their IQ2-class | 2.25 | 34.0 GB | **no** — no GGUF path in our tree or the pin |
| their IQ3-class | 3.06–3.44 | 47–52 GB | **no** |
| int3, group-128, fp16 scale | 3.13 | 47.2 GB | **no** — nothing instantiates num_bits=3; prebuilt `.so` unverified |
| NVFP4 (ours, experts) | 4.5 (4.83 with MTP) | 68.0 GB | **yes**, one W4A16 kernel serves all M |
| block-128 fp8 (ours, dense) | 8.25 | — | **yes** |

Declared, not measured: our NVFP4 kernel exists at prefill but is not
prefill-tuned. Availability is not performance, and this table only claims
availability.

**The 5–6x decode deficit against the reference is retracted** (2026-11-16,
re-measured from existing traces on both stacks, de-clipped by anchoring on a
repeating annotation and spanning first-anchor to last-anchor so both ends sit in
the same phase of a step). Their expert bucket had been **under-counted ~4.5x**
because the attribution covered only one piecewise-graph annotation — 22% of their
kernels. Same window, same TP degree, same NVFP4 format, the real gap is **1.59x
per launch on gate_up, 1.87x on down, 1.70x per step**, and the expert bucket is
**19.1% of a decode step**. So matching their per-launch times buys **1.13x**, and
deleting the bucket entirely buys 1.33x.

**That bucket cannot carry the per-card ratio** — which is §1's conclusion reached
from the other direction, by two peers with different methods.

**Dense term, and we win it.** Their non-expert tensors are fp16 at 16 bit/param;
ours are block-128 fp8 at 8.25 — **~1.94x smaller**. Their capacity advantage
therefore cannot come from dense weight format. It has to come from the expert
term and placement. **And the smaller format is free rather than paid for:**
removing fp8 from the dense path measured **9.3% slower per decode step** and
prefill-neutral, so our dense bytes cost us nothing in speed either.

## 4. L2 — placement, re-priced at the measured residency

The host-tier cost is a function of the **miss** fraction, so the hit rate decides
whether placement is a bargain or a tax:

| regime | miss fraction | bytes/token off host | at 44.14 GB/s |
|---|---:|---:|---:|
| header default, held-out `h=0.6447` | 35.5% | 235.8 MB | **5.3 ms** |
| auto-sized, measured `h=0.985` | 1.5% | 10.0 MB | **0.22 ms** |
| **realised steady state, `h=0.989`→`h=0.997`** | 1.1%→0.3% | 7.3→2.0 MB | **0.17→0.05 ms** |

The realised row is confirmed by an independent channel: **1 PCIe expert read out
of 32,484 routed** (0.0%), and the process's `read_bytes` rose **7.7 MB in total**
across a 4,148-token prefill plus 64 decode tokens. So the host tier is a
**cold-start cost only** — 69.3 GiB streamed at load — and in steady state it is
not measurable at token granularity.

That settles the gross-versus-residual question the other way from the header's own
costing: **663.6 MB/token is gross demand**, confirmed independently as 48 × 10 ×
~1.44 MB ≈ 691 MB, and the residual is ~0 in this configuration.

So at the realised residency the host bus is **at most 0.17 ms on a 17.9 ms
token** — 1% of the step, and not the constraint. **VRAM capacity at 80.2%
residency is the constraint**, and the 5.3 ms figure describes a configuration
nobody is running. The earlier framing —
"capacity paid for in latency" — is wrong at 32 GB.

What this does to our side is a **conflict, not a gap**: we already run a host
tier (PLE, 47.75 GiB, pinned, off the card), and the generic knobs that would put
experts on one are **mutually exclusive with PLE offload**
(`srt/arg_groups/memory_hook.py:30-36` — generic layer offload stages the pinned
PLE embedding back to the device). So expert placement for us is a choice
between two host tiers, not a flag.

## 5. Overheads we may be paying without noticing

| term | value | class |
|---|---|---|
| our resident weights per card | **22.14 GiB allocated / 22.37 reserved** | measured, `restomod-load-memory` |
| duplicate prepared weight layout | **0** — `prepare_sm70_turbomind_fp8_linear` assigns the prepared tensor to `layer.weight`; the pre-repack tensor is a local that dies at return. The 0.70 GB/rank duplicate measured earlier is QPN8's packed codes, default-off since #306 | **refuted**, source-read |
| PLE host tier | 47.75 GiB pinned | measured |
| hicache | 16 GB host | configured, not measured |
| mamba state floor pinning mem-fraction | ~0.67–0.82 | measured floor |
| replicated `hyper_connection` | 1.19 GiB = ~44% of per-rank dense | derived |
| graph pool bytes | **unmeasured in both stacks** | absent from both logs; not inferred from capture counts |

The duplicate-layout row came back **refuted**, which removes a term I had flagged
as possibly as large as the quantization gain it was meant to buy: the largest
suspected overhead is not there.

### 5.1 The decode census: a retracted figure, and a partial capture

**A first reading of this window put collectives at 24.7% of the decode step and
made them the largest single family. That is retracted: the shares were computed
over the whole capture, which contains two 4,096-token prefill spans (1,536 ms) and
two draft spans despite `profile_stages=["decode"]`.** Restricted to the 40
`step[VERIFY bs=1]` spans:

| quantity | value |
|---|---|
| whole capture | 1,389.5 ms / 34,107 kernels |
| verify spans, wall | 522.5 ms |
| **kernels inside verify spans** | **238.3 ms / 23,897** |
| `all_reduce_1shot` | 983 calls × **8.1 µs** = 7.97 ms = **~3.3%** of verify kernel time |
| — per captured step | **0.199 ms** |
| `decode_moe_gemv` | 480 calls × 61.4 µs |
| `sm70_marlin_moe` | 480 calls × 43.6 µs |

**So collectives are not the decode lever**, whatever the prefill-weighted table
said: 0.199 ms on a 24.7 ms step. My earlier "~64 µs per hand-off" came from reading
the **12** `ncclDevKernel_AllReduce_RING` instances (63.0 µs) as if they were all of
them; the 1-shot path dominates at 8.1 µs.

**That does give the first evidence that custom allreduce is active** — the 1-shot
kernel is the custom path, and it is what nearly every collective uses. It is not
yet a clean answer: **12 RING calls in the window still need an explanation**, and
the suite reports failure-absent-only, so "no failure line" was never evidence
either way.

**Directional, not a ratio: their step spends ~15 ms in communication where ours
spends under one.** On their trace two cross-device reduce kernels take **10.96
ms/step** plus **4.05 ms of memcpy**; on ours a single 1-shot all-reduce takes
**0.61 ms**. The mechanism is the point — two cross-device reduces plus a copy
against one fused allreduce — and it puts the communication cost on their side of
the ledger. **It must not become "we win comm by 20x":** different stacks,
different models, and **collective durations are the least trustworthy numbers a
profiler produces**, with spin-wait inflated on both sides. Direction and
mechanism are recorded; the magnitude is not.

**The capture is partial, so these shares bound the order of the families and
nothing else.** 12 GEMV + 12 Marlin calls per step against ~36 expert layers, and
24.6 collectives per step against ~96 expected: roughly a third of the work is
captured. **A fix's savings cannot be computed from this window.** That needs either
a complete capture or the counter:

> one `int[4]` incremented by `n` inside `maybe_call` and inside the Marlin
> dispatch, printed at teardown — no profile window, and it answers which shapes
> each expert path serves, which is the question the partial capture cannot.

**Rule adopted from this:** an all-reduce count is quoted **with its per-call size,
its TP degree, and its span filter**. A family share from a window that contains
prefill is not a decode share, and "96 collectives" alone cannot be compared between
TP2 and TP4.

## 6. Transfer to bluewhale

Config (cited): 40 layers, 384 routed experts, top-6, `moe_intermediate_size`
2304, hidden 5120, 64 heads, **1 KV head**, `head_dim` 512,
`qk_rope_head_dim` 64, vocab 129280, 1M context via yarn.

| term | transfers? | why |
|---|---|---|
| L1 expert bytes | **yes, and it dominates harder** | 271.8 G weight params / 288.8 GB with scales (§6.1) against Flash-Next's 120.8 G |
| L2 resident fraction | **yes, in mechanism** | but 384 experts × 6 active is a different routing load from 512 × top-k: **redo the math, do not scale it** |
| KV bytes/token | **model-specific** | MLA versus GQA is architectural; ~46 KB/token at fp16 (§6.2), but not the largest term |
| **embedding table, 204 GB** | **yes, and it changes tier** | a sparse gather of ~530 B to ~2.1 KB per token from a 204 GB table — an **I/O term, not a VRAM term** |
| PLE, hicache, mamba floor, `hc` replication | **no** | Flash-Next only; re-derive per model |

### 6.1 Measured class split, and three expert numbers that do not reconcile

All 48 shards' headers parsed — **510.29 GB of tensor data**, matching the file
count, so this is the whole checkpoint:

| class | bytes | params | bits/param |
|---|---:|---:|---:|
| **expert** | 296.00 GB | 296.00 G | 8.00 |
| **embed** | **204.15 GB** | 203.45 G | 8.03 |
| attn + ffn other | 7.22 GB | 7.12 G | 8.11 |
| misc | 2.10 GB | 1.20 G | 13.95 |
| vision | 0.82 GB | 0.41 G | 16.00 |

**The embedding table is 204 GB — 40% of the checkpoint — and it is a sparse
lookup, not a per-token cost.** At `vocab_size` 129,280 × `hidden_size` 5120 the
token embedding is 0.66 G params, so this is a different object. The config names
it: an **Engram** table on two layers — `engram_layer_ids [1, 14]`,
`engram_num_embeddings [384006168, 384016682]`, `engram_n_heads 8`,
`engram_head_dim 256`, `engram_max_ngram_size 4`. 768.02 M entries in 203.45 GB is
**~265 B per entry**, i.e. 2048 values at about one bit.

Per token that is **one entry per Engram layer ≈ 530 B**, or — since
`max_ngram_size` 4 implies up to four n-gram lookups per layer — **≈ 2.1 KB** as
the prudent ceiling. Either figure is irrelevant beside the **4.51 GB/token**
expert demand below, so the table moves **out of the VRAM resident set and into an
I/O term**. It is the same shape we already run as PLE host-offload, larger and
colder; its placement is a tiering decision, not a capacity one.

**Expert bytes, resolved per tensor.** One expert, every tensor:

```
w1  I8 [2304, 2560]  5,898,240 B      w1.scale F8_E8M0 [2304, 160]  368,640 B
w2  I8 [5120, 1152]  5,898,240 B      w2.scale F8_E8M0 [5120,  72]  368,640 B
w3  I8 [2304, 2560]  5,898,240 B      w3.scale F8_E8M0 [2304, 160]  368,640 B
```

**17,694,720 weight params + 1,105,920 scale bytes = 18.80 MB per expert**, and
× 15,360 pairs = **288.8 GB including scales** (weights alone 271.8 GB at 8.00
bit/param). The config-arithmetic reading of 543.6 G is **deleted**: 2304 is w1's
*output* and 2560 its stored input, so there is no 2-way K-split and no doubled
union — that came from assuming `hidden_size` 5120 is the expert input width.

**The 7.22 GB residual is a third namespace, not a double-count — checked and
refuted.** Summing **unique tensor names** gives **96,085 names, 0 duplicates,
510.29 GB**: the checkpoint is complete and nothing is counted twice. The expert
class decomposes exactly:

| group | bytes |
|---|---|
| 40 main layers × 7.219 GB | **288.76 GB** |
| 3 MTP layers × 2.406 GB | **7.22 GB** |
| **expert class total** | **296.00 GB** |

Per-layer expert bytes are **uniform across all 40 main layers**, so layer 6's dump
is representative rather than a sample. The extra group is `mtp.0/1/2.ffn.experts`,
768 tensors each.

**So every expert figure has to state its MTP convention**, because the served
model keeps the draft resident: **18.80 MB per expert** is a main-layer figure,
**288.76 GB** the main-layer total, **296.00 GB** all-in. This is the same class
boundary that reconciled the Flash-Next map (63.28 + 4.69 = 67.97 GiB). On both
models the MTP question must be answered before a byte total means anything — and
the first attempt at this page's own arithmetic got it wrong in the other
direction, by omitting the draft entirely.

**Both formats still need an offline pass.** Dense is block `[32,32]` ue8m0 and
`can_use_sm70_turbomind_fp8` refuses anything but `[128,128]` (a refusal, not a
fallback); the shipped expert tensors are INT8 with ue8m0 block-16 scales, which
is neither our Marlin NVFP4 nor the TurboMind fp8 path. And `expert_dtype: fp4`
in the config **contradicts the tensors** — it is not a byte coefficient.

### 6.2 KV, and whether tensor parallelism helps at all

512 + 64 = **576 values per layer**, × 40 = 23,040 values = **46,080 B/token at
fp16 (45 KiB)**, 23,040 B at fp8. At the full 1M context: **46 GB fp16 / 23 GB
fp8**, i.e. past a 32 GB card on KV alone.

With **one KV head**, head-wise tensor parallelism gives no KV relief. Whether
the 512-value latent shards along its own dimension depends on an MLA path that
**our tree does not have** — `unverified`, and it decides whether TP4 buys 4x KV
headroom or 1x. That single fact moves the long-context answer by 4x and is the
first thing to check, ahead of any weight-format decision.

### 6.3 The dense file

`weight_block_size: [32, 32]`, `scale_fmt: ue8m0`; our block-FP8 path requires
`[128, 128]` and refuses anything else. In bytes the difference is trivial —
8.008 bit/param against our 8.25 — but in usability it forces one of two offline
passes, because **no unquantized checkpoint is published**:

- **requantize dense to block-128** — bytes unchanged, one offline pass, our
  kernels run;
- **dequantize dense to 16 bit** — **2x the dense bytes**, no requantize.

`expert_dtype: fp4` does **not** match the shipped experts (§6.1), so **both**
files need an offline pass, not just the dense one.

Confirmed at source: `can_use_sm70_turbomind_fp8` returns False unless
`weight_block_size == [128, 128]`, so the fp8 path never takes the layer — a
refusal, not a fallback that might still work. And the dense side really is block
32: the shared expert's scale is `[72, 160]` on a 5120-wide weight, i.e.
5120/160 = 32.

### 6.4 The transfer: the mechanism transfers, the viability does not

Using §6.1's 18.80 MB per expert:

| quantity | Flash-Next | bluewhale |
|---|---|---|
| (layer, expert) pairs | 24,576 | 15,360 |
| bytes per expert | 1.38 MB | **18.80 MB** |
| slots in the same 26.50 GiB cache | 19,721 | **~1,513** |
| **resident fraction** | **80.2%** | **~9.8%** |
| gross expert bytes/token | 663.6 MB | **4.51 GB** |
| host bytes/token at that residency | ~0 | **~4.07 GB** |
| host cost at 44.14 GB/s | ~0.05 ms | **~92 ms** |
| fully resident, expert reads at 900 GB/s | — | **~5.0 ms/token floor** |

**Arithmetic correction to the figure this was first built from:** "6 active
experts is ~116 MB/token" omits the layer multiplier. Routing is top-6 **per
layer**, so the gross is 40 × 6 × 18.80 MB = **4.51 GB/token** — the same
construction that gives Flash-Next 48 × 10 × 1.38 MB = 663.6 MB. The 116 MB figure
understates the host tier by 40x and would have made it look viable.

So:

- **The host tier is not a deployment option here.** At ~9.8% residency it costs
  ~92 ms/token against a 17.9 ms/token decode — two orders of magnitude, not a
  tuning knob.
- **Standing still is not an option either.** Fully resident, expert reads alone
  are ~5.0 ms/token at 900 GB/s: a floor before dense, attention or KV.
- **The expert term is binding, and it is where quantization lives.** 288.8 GB at
  the base checkpoint; **77.5 GB at 2.28 bit** — the difference between roughly ten
  cards and three at full residency. For bluewhale, quantization is not a lever, it
  is the only term that moves the card count.
- **The embedding table is not a capacity term.** ~530 B to ~2.1 KB per token off a
  204 GB table is an **I/O** question, and the coefficient that decides it is
  random-read latency on the real SSD at that granularity, plus whether a warmed
  chunked-prefill batch coalesces the lookups (4,096 tokens × 2–8 entries = 8k–33k
  reads that can batch). If they coalesce, NVMe costs nothing and the table never
  needs residency; if latency dominates, the answer is host RAM — 204 GB exceeds
  most hosts — or prefetch depth.
- **Kernel availability stays gating** for the expert term, which is now the one
  that decides the deployment shape.

So the transferable mechanism is **a sparse table tiered outside VRAM, plus expert
residency in VRAM** — both of which we already run in a smaller form: PLE offloaded
at 47.75 GiB, experts resident at 4.5 bits. What does not transfer is the ratio. On
Flash-Next the table is 47.75 GB and host-resident, and quantization gets the experts
under one card. On bluewhale the table is 4.3x larger and belongs on NVMe, while the
experts need ~10 cards at 8 bits and ~3 at 2.28. Same mechanism, different numbers —
which is the answer the transfer question was actually asking for.

## 7. Pending

| item | owner | state |
|---|---|---|
| **a complete decode census (this one covers ~1/3)** | kernel-crafter | **open, next** — shares bound order only; the counter answers the shape split |
| **the 12 RING all-reduce calls** | kernel-crafter | **open** — the 1-shot path is active, so these are an exception to explain, not a fallback |
| `f_res` realised + policy | multi-node | **answered** — 80.2%, §2.1; prefetch/eviction not printed |
| duplicate prepared copy | coordinator | **refuted** — §5 |
| GQA KV layer split | kernel-crafter2 | **dropped** — not the binding term |
| bluewhale expert bytes | coordinator | **resolved per tensor** — 18.80 MB/expert, 288.8 GB (§6.1); the 7.22 GB residual was three MTP layers, not a double-count |
| bluewhale Engram table tier | — | per-token bytes answered (~530 B–2.1 KB); **open**: SSD random-read latency at that granularity, batched under chunked prefill |
| does the IQ2_XS repack ship the Engram table at all | multi-node | **open** — 155 GB total against 204 GB of table alone, so it must quantize it far below 8 bits or omit it; one per-tensor dump settles it |
| MLA latent sharding under TP | — | **open**; moves the long-context answer by 4x (§6.2) |
| per-card throughput attribution | kernel-crafter2 | **attributed to capacity and accept length, not kernels**: expert GEMM is 19.1% of a step at 1.70x, ceiling 1.13x, and our kernel volume is 21.6 ms against their 22.9. `docs/runs/2026-11-16-decode-gemm-gap.md` carries the superseded 5–8x and needs a correction |
| kernel availability per expert format | kernel-crafter | ladder delivered; gating for bluewhale |

### 7.1 Corrections in the record, 2026-11-16

Five numbers moved today. They are listed together so a reader of the run notes
sees the churn rather than a tidy history, and so no one quotes a superseded value
from an older message.

| claim | status | what it was |
|---|---|---|
| decode expert deficit **5–6x** per call | **retracted → 1.70x per step** | attribution covered one piecewise-graph annotation holding 22% of the reference's kernels, so their bucket was under-counted ~4.5x. A denominator error, not a near miss. |
| collectives **24.7%** of the decode window | **retracted → 0.199 ms/step (~3.3%)** | the shares came from a capture holding two 4,096-token prefills despite `profile_stages=["decode"]` |
| **7.22 GB** double-count in the expert class | **refuted → three MTP draft layers** | my hypothesis; the unique-name sum is 96,085 names, 0 duplicates, 510.29 GB |
| **116 MB/token** of expert host traffic | **retracted → 4.51 GB/token** | the gross figure omitted the per-layer multiplier: top-k is per layer |
| **543.6 G** expert parameters | **deleted → 271.8 G weights, 288.8 GB with scales** | `2304` is w1's output; assuming `hidden_size` 5120 was the expert input invented a 2-way K-split |

Two of the five were mine. Both are the same failure: a number that reconciled
with something nearby, so it was not checked against the artifact.

## 8. What this page does not say

- It does not claim a measured throughput delta against the competitor. Their
  experts are a different format and a different amount of work per token, so no
  tok/s ratio across the two is evidence here.
- It does not say a given bit width is where we should land. Quality is out of
  scope; kernel availability is recorded only because a format with no SM70 kernel
  costs the whole model.
- **It does not quote a bluewhale expert byte figure.** Three readings disagree
  (§6.1), the 8.00 bits/param is derived rather than measured, and the table stays
  unbuilt until the repack's own tensor list settles it.
- It does not model the 204 GB embedding table's per-token traffic — the single
  item that decides whether bluewhale is deployable on 32 GB cards at all.
