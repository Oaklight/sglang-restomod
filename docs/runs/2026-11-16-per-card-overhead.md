# 2026-11-16 — Per-card GPU overhead: measured, and it is 3.5x the planning constant

Status: **measured**. Four components, each reported separately, on lambda4 GPUs 4-7.
This sets the capacity budget: the planner's `other = 3.6 GB` is not what our stack
pays.

**No receipt covers this run.** It is a measurement on an already-receipted tree, and
every flag is recorded below so the configuration can be reproduced.

> **Superseded later the same day: the shipped launcher measures 7.17 GiB as 5.11 GiB.**
> The arm this page reports was my own wrapper script. Run again through the shipped
> `launch-flashnext.sh` — unchanged, on the same four cards — the same decomposition
> gives a fixed footprint of **5.11 GiB (5.49 GB)**, and a residual after naming the
> mamba cache of **2.72 GiB**, not 4.78. Everything below stands as the record of the
> wrapper arm and of the method; the numbers to use are in
> [the shipped-configuration table](#the-shipped-configuration-measured) at the end. The
> four-card conclusion still holds, but the gap is 8.9 GB rather than 17.3.

## Result

Per card, mean of four ranks, `nvidia-smi` used at steady state with the server
having served a request:

| | graphs on | graphs off | delta |
|---|---:|---:|---:|
| used, mean of 4 ranks | **30,407 MiB** (29.70 GiB) | 29,692 MiB (29.00 GiB) | **716 MiB** |
| per rank, GPUs 4-7 (on) | 30,674 / 30,284 / 30,356 / 30,316 MiB | 29,958 / 29,568 / 29,640 / 29,600 MiB | |

Decomposition of the 29.70 GiB per card:

| term | value | how obtained |
|---|---:|---|
| weights, torch reserved | **22.37 GiB** | engine's own line: `restomod-load-memory: allocated=22.14 reserved=22.37` |
| KV cache | **0.153 GiB** | engine's line: 172,032 tokens, K 0.08 GB + V 0.08 GB — **identical in both arms**, so the A−C delta is the graph pool alone |
| CUDA context | **0.357 GiB** (366 MiB) | separate process that only initialises CUDA; `nvidia-smi` delta **366 MiB on lambda6 and on lambda4, identical on all ranks** |
| CUDA graph pool | **0.699 GiB** (716 MiB) | graphs on minus graphs off, same mem-fraction, same KV |
| workspaces + allocator slack | **6.11 GiB** | residual, **reported as a sum** — it could not be split further without instrumenting the engine |

**Fixed footprint excluding weights and KV: 7.17 GiB per card (7.70 GB).** That is
**3.5x the 2.2 GB target** and **2.1x** the planner's 3.6 GB NVIDIA constant, which
is the number the 1.75 bpw plan was built on.

**A structural point that fell out:** with `--mem-fraction-static 0.82`
(26.9 GiB of 32 GiB) the engine had almost no room left for KV — it allocated
**172,032 tokens in 0.16 GB**, about 0.93 KB/token. Our context length is limited by
the static footprint, not by the KV budget. At 22 GiB/rank of weights plus 7.70 GB of
fixed cost, the card is 90% spoken for before a single token is cached.

## What it does to the 4-card plan

Same budget shape as the comparison page, with the measured overhead substituted:

| cards | expert capacity at our measured overhead | 1.75 bpw (117.42 GB) | 2.25 bpw (149.06 GB) |
|---|---:|---|---|
| 4 | **100.1 GB** | short **17.3 GB** | short 48.9 GB |
| 5 | 125.1 GB | **fits, 7.7 GB spare** | short 24.0 GB |
| 6 | 150.1 GB | fits, 32.7 GB spare | fits by **1.0 GB** — too tight to ship |
| 8 | 205.7 GB | fits, 88.3 GB spare | fits, 56.7 GB spare |

**So the four-card plan is dead and the five-card plan is the cheap one.** The
1.75 bpw set needs **5 cards**, not 4; the 2.25 bpw set needs **6 with 1 GB to spare,
or 7 to be safe** — not the 6 the earlier projection gave, because that projection
used 3.6 GB of overhead and we pay 7.70 GB.

The gap is not the expert set and not the card-count formula. It is 4.1 GB per card of
overhead nobody was counting, and 6.11 GiB of it sits in one unattributed residual.

## The residual is the actionable part

6.11 GiB per card of workspaces and allocator slack is 8.7x the graph pool and 17x the
CUDA context. Candidates, **none separated**:

- Marlin, Triton and TileLang scratch buffers, sized per shape and never returned;
- prefill activation buffers sized for `--chunked-prefill-size 4096`;
- the draft model's buffers under `SPEC=1` — a second model's worth of scratch;
- CUDA caching-allocator fragmentation. `reserved − allocated` for weights is only
  0.23 GiB, so the pool holds the rest rather than returning it to the driver.

**Each GiB returned here is a GiB of experts that fits**, which is worth more than any
kernel change in the decode thread. Cheapest experiments, in order:

1. `--chunked-prefill-size 2048` and re-measure.
2. `SPEC=0` and re-measure; the difference is the draft's static cost.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to separate fragmentation from
   real allocation.

## Configuration, exactly as run

- tree: SGLang `.build/sglang` **v0.5.23-dirty**; restomod **ff2d0dc72b**; Marlin
  `_C.abi3.so` md5 **49458692777d78ed6dd44a10153cc09f** (matches the protocol's
  recorded hash).
- `--tp-size 4 --ep-size 1 --moe-runner-backend marlin --attention-backend qsa_v100
  --quantization modelopt_fp4 --dtype float16 --page-size 64 --mem-fraction-static
  0.82 --max-running-requests 8 --chunked-prefill-size 4096 --context-length 65536
  --enable-hierarchical-cache --hicache-size 16 --ple-offload-embedding`
- `SPEC=1`: NEXTN, 3 steps, topk 1, 4 draft tokens, rejection sampling. Decode CUDA
  graphs captured for bs 1-8; **prefill graphs disabled**.
- Arm A: graphs on, port 31998. Arm C: `--disable-cuda-graph`, port 31997. Both at
  0.82, both served a `/generate` request before the reading, both stopped by
  **process group** and the cards returned to 4 MiB.
- **Both arms report the TurboMind FP8 ops path unset**, so `fp8_lm_head` and
  `fp8_dense_quant` logged "unavailable, skipped" and dense stayed bf16. The fixed
  overhead does not depend on that choice, but this arm is **not byte-identical** to
  the served 126.2 tok/s configuration and must not be described as such.

## Notes for whoever runs this next

- **The operator's home directory is not writable inside the container on this host**
  — a lambda6 arm died with `OSError: [Errno 28] No space left on device` in the
  SGLang JIT build under the JIT cache beneath `$HOME`. `HOME` does not cross the
  singularity boundary unless passed as `--env HOME=...`; the launcher now passes it,
  plus `XDG_CACHE_HOME`.
- **lambda6's root filesystem is 100% full** (1.7 T of 1.7 T, 0 available), so the
  shipped recipe's scratch directory under `/tmp` fails before the model loads. lambda4 has
  1.2 TB free and works.
- Both arms were single-variable: KV came out identical at 172,032 tokens, so the
  716 MiB delta is the graph pool and nothing else.

## The shipped configuration, measured

Three arms, same four cards, same model, one factor each. "Fixed" excludes weights and
KV; the graph pool is borrowed from the graphs-on/off pair, which was measured in arm A
only.

| arm | what it is | used, mean | KV tokens | mamba | residual after naming | **fixed** |
|---|---|---:|---:|---:|---:|---:|
| A | my wrapper, no `gdn_tilelang_sm70`, TurboMind unset | 30,407.5 MiB | 172,032 | 1.43 GB | 4.78 GiB | 7.17 GiB (7.70 GB) |
| **S1** | **`launch-flashnext.sh` as shipped** | **28,295.5 MiB** | 172,032 | 1.43 GB | **2.72 GiB** | **5.11 GiB (5.49 GB)** |
| S2 | S1 + `SGLANG_SM70_TURBOMIND_OPS_PATH` (dense FP8 engaged, 416 layer binds) | 28,409.5 MiB | 183,168 | 1.48 GB | 2.77 GiB | 5.20 GiB (5.59 GB) |

All three served a `/generate` request before the reading, and the per-rank spread is
under 80 MiB inside every arm.

Three things this settles:

1. **The residual is 2.72 GiB in the configuration we ship**, and dense FP8 does not
   move it (2.77 GiB; the difference is the engine spending the freed room on more KV
   tokens and a larger mamba cache, `max_mamba_cache_size` 72 against 68). The 4.78 GiB
   in arm A is a property of *that wrapper*, not of the stack, and the shipped residual
   is not a bf16-dense artifact.
2. **`Mamba Cache is allocated` is a named term, not part of the residual**: conv 0.04 +
   ssm 0.91 + intermediate_ssm 0.47 + intermediate_conv 0.01 = **1.43 GB per card**
   (1.33 GiB). It is a Flash-Next term: a model with no linear-attention layers pays
   none of it.
3. **KV is bounded by the static footprint, and the engine agrees with the card**:
   `available_gpu_mem` is 5.81 GB in S1 and 5.68 GB in S2, matching `vram - used` to
   within 0.2 GB in both. The residual is a real allocation, not a measurement artifact.

### Capacity with the shipped number

`n x (34.36 - 1.645 - fixed)`, fixed taken at the larger shipped arm (5.59 GB):

| cards | expert capacity | 1.75 bpw (117.42 GB) | 2.25 bpw (149.06 GB) |
|---|---:|---|---|
| 4 | 108.5 GB | short **8.9 GB** | short 40.6 GB |
| 5 | 135.6 GB | **fits, 18.2 GB spare** | short 13.4 GB |
| 6 | 162.8 GB | fits, 45.3 GB spare | fits, 13.7 GB spare |

Four cards would need the fixed footprint down to **3.36 GB**, i.e. 2.2 GB out of 5.59 —
of which 1.48 GB is a mamba cache sized by `max_mamba_cache_size` and the remainder is
the residual.

**This arithmetic is Flash-Next's.** A different model substitutes its own `fixed`; the
formula and the four measurements that feed it are in
[the instrument](2026-11-16-overhead-bench-spec.md).

### What is still not attributed

The 2.1 GiB by which arm A exceeds S1. The candidates are the `gdn_tilelang_sm70`
component (registered in S1 and S2, absent from A) and the singularity home flag A
substituted for `--home`. `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` is not a candidate:
`bootstrap.py:435-449` only chooses between raising and warning, it allocates nothing.
Not chased, because it does not change which configuration we ship.
