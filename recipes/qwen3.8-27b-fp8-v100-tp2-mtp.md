# Qwen3.8-27B-FP8 on 2x V100-SXM2, TP2, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-27B-FP8` (block FP8, hybrid dense + GDN) |
| Hardware | 2x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP2, no EP |
| Context | 32,768 to 262,144 (measured at both) |
| Engine | Two stacks, and the text says which rows are which. The fork tables were measured on `dg1kjd/sglang-sxm2` @ `73abfe1` plus the lab patch set (`patches/sglang-sxm2/` in `Oaklight/lambda-v100-lab`). Every figure marked "at the pin", and every component row, was measured on `sgl-project/sglang` `v0.5.22` @ `baa9befa` with this layer installed. |
| Components | `attention_v100`, `fp8_dense`, `fp8_lm_head`, `spec_sm70`, `gdn_tilelang_sm70` |
| Date | 2026-11-02 (KV dtype: e4m3 at the fork, fp16 at the pin — see "KV dtype") |

## Flags

```
--dtype float16 --quantization fp8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
# --kv-cache-dtype fp8_e5m2           # optional 1-byte cache for 256k; equal
                                      # decode at 32k since #129. See "KV dtype".
--tp-size 2 --context-length 262144   # max_position_embeddings
--mem-fraction-static 0.85 --max-running-requests 4
--chunked-prefill-size 1024 --max-prefill-tokens 8192
                                      # one chunk is one decode stall; see "Prefill chunks"
--disable-prefill-cuda-graph          # decode graph stays on
# MTP only up to ~48k: see "Pick the spec setting by context" below.
# SGLANG_27B_SPEC=auto in the launcher does exactly this.
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--sampling-backend pytorch                    # see "Non-greedy requests"
--speculative-use-rejection-sampling          # see "Non-greedy requests"
--speculative-draft-attention-backend tilelang_fa_v100  # see "The draft backend"
--triton-attention-num-kv-splits 40    # draft decode: 2 KV heads x 40 = 80 CTAs
```

Add `SGLANG_V100_DRAFT_EXTEND_NATIVE=1` to the environment to put the draft
extend on the same grouped split-KV kernel the target verify uses. It is the
single largest decode change on this model at long context:

| prompt tokens | MTP decode, flag off | flag on |
|---|---:|---:|
| 8,204 | 42.99 / 46.16 tok/s | **60.40 / 64.50** |
| 31,756 | 17.89 / 17.86 tok/s | **41.48 / 44.71** |

Greedy, 128 output tokens, two reps, lambda12, accept length 2.73-3.12 in both
arms. Prefill is unchanged. The flag defaults off because the CUDA graph capture
path still routes a draft extend to Triton; on this model the draft extend runs
eagerly, so the capture path is not exercised. Receipt
`evidence-e5a5740-draft-extend` covers the sampled run at the same commit.

**Three draft steps, not two or four.** With the draft extend fixed, the draft
phase is the largest term in a step, so the depth was swept at 31.1k with 40 KV
splits, 128 output tokens and four reps:

| steps / draft tokens | accept length | decode tok/s |
|---|---|---|
| 2 / 3 | 2.40-2.60 | 48.69 |
| 3 / 4 | 2.88-3.50 | **53.37** |
| 4 / 5 | 2.90-3.92 | 52.11 |

Two steps saves a third of the draft cost and gives up 22% of the accept
length; four steps buys 0.06 of accept for one more full draft step. Keep
`--speculative-num-steps 3 --speculative-num-draft-tokens 4`.

The 4/5 row was also the only one that needed `--mem-fraction-static 0.80`: at
0.85 the launch dies while loading weights, on a 136 MiB allocation with 117 MiB
free.

**With an fp8 KV cache the optimum moves to five.** The table above is fp16 KV
with 8 splits. On `fp8_e5m2` with #129's native draft extend the draft step is
cheaper, so depth is worth more; the same sweep at 31.7k with 40 splits, four
reps per arm and eight for the two that matter:

| steps | decode tok/s, mean of 4 | mean of 8 | median of 8 |
|---:|---:|---:|---:|
| 3 | 56.32 | 58.41 | 58.56 |
| 4 | 61.07 | — | — |
| 5 | 66.06 | **62.87** | **63.40** |
| 6 | 60.10 | — | — |

Five steps is +7.6% on the mean and +8.3% on the median against three, with
prefill unchanged at 1,633-1,656 tok/s in every arm. The per-rep spread is
10-15 tok/s in both arms, so read the medians.

It costs memory for the captured graphs: at 0.85 only 3 steps start (4 steps
OOMs on a 136 MiB allocation), 5 steps needs `--mem-fraction-static 0.80`
(263,952 tokens of pool) and 6 needs 0.78 (237,856).

**And it stops paying past 64k.** Every draft row reads the whole prefix, so a
step costs `prefix` bytes; the same launch at 5 steps measures 34.80 / 32.86
tok/s at 130k against 36.19 / 29.66 at 3 (inside the spread), and 18.54 at 260k
against 20.24. Capacity is not the reason — 5 steps at 262k context and mem
0.82 starts with 280,832 tokens of pool — the per-step cost is.

So: `--speculative-num-steps 5` with `--kv-cache-dtype fp8_e5m2` up to 64k, 3
steps from 128k up, and 3 on fp16 at any context.

The engine raises the draft-token count itself. It logs
`speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when
speculative_eagle_topk == 1`, so `--speculative-num-steps 5
--speculative-num-draft-tokens 4` runs as 5 steps and 6 draft tokens, and
`/get_server_info` reports 6. A 3-step launch asks 4 and stays at 4. Quote the
engine's number: 3/4 and 5/6.


**Non-greedy requests need three things at the pin.** Without them the server
either refuses to start or dies on the first temperature-1 request:

1. `spec_sm70` in `SGLANG_RESTOMOD_COMPONENTS` and
   `--sampling-backend pytorch`. Otherwise a temperature-1 request reaches
   FlashInfer's sampler, which refuses SM70:
   `RuntimeError: FlashInfer requires GPUs with sm75 or higher.`
2. `--speculative-use-rejection-sampling`. Otherwise EAGLE verification builds
   `sgl_kernel_jit_speculative_sampling`, and that build fails against CCCL:
   `flashinfer/sampling.cuh(40): error: namespace "cuda" has no member "maximum"`.
3. `--speculative-draft-attention-backend triton`. Otherwise MTP does not start
   at all: `ValueError: EAGLE is not supported in decode attention backend
   tilelang_fa_v100`, because `speculative/draft_utils.py:119` resolves the
   draft backend by name from a map that does not hold `tilelang_fa_v100`.

Greedy requests reach none of these paths, which is why a temperature-0
benchmark passes while a default-sampling request kills the server.

**The draft backend.** `triton` also works, but it puts the draft extend on the
Triton extend kernel, which assigns one CTA per head and walks the whole prefix.
At 32k that kernel is 96.7 ms of a 168 ms step; the target verify costs 10.7 ms
(issue #75). `tilelang_fa_v100` is selectable since pull request #99, and with
`SGLANG_V100_DRAFT_EXTEND_NATIVE=1` the draft extend moves off that kernel
entirely.

**Draft decode attention: the split count.** With the draft extend on the native
kernel, the largest term left in a draft phase is the three draft steps'
attention, on the Triton split-KV kernel at grid `(1, 2, 8)`: one row, 2 KV heads
and 8 splits, 16 CTAs on an 80-SM card. `--triton-attention-num-kv-splits` sets
it:

| splits | grid | CTAs | draft attention | draft phase | decode at 31.1k |
|---:|---|---:|---:|---:|---:|
| 8 (default) | `(1, 2, 8)` | 16 | 19.43 ms | 25.14 ms | 43.73 tok/s |
| 20 | `(1, 2, 20)` | 40 | 12.73 ms | 18.20 ms | 44.99 tok/s |
| 40 | `(1, 2, 40)` | 80 | 9.91 ms | 15.14 ms | **50.75 tok/s** |

128 output tokens, four reps per arm, fp16 KV, TP2, 31.1-31.7k prompt. 40 splits
is 2 KV heads times 40, one CTA per SM. The target verify is the control: it
keeps its own kernel and computes its own split count, and measures 17.9 to
18.3 ms per step in all three runs.

The split count changes the reduction order, so it changes the text. Three
31,744-token prompts over 96 greedy tokens measure a mean absolute
log-probability shift of 0.0016 nats against the default (rms 0.0039, worst
0.0304) and no token divergence in 288; one near-tie flipped on a different
prompt. The KV dtype change already in use measures 0.0383 to 0.0627 on the same
metric. Keep the flag off jobs that need byte-identical output, and measure
below ~8k before making it the default.

That last measurement is done: at 2,019 prompt tokens the two arms measure 69.69
against 71.97 tok/s mean, at 4,023 63.58 against 66.26, with the spreads
overlapping completely and prefill identical in every row. Below 8k the flag is
a no-regression, not a gain; its measurable win is at 32k.


The trace, the three profiles and the log-probability comparison are in the lab
note `notes/mtp-long-context-attribution.md`.

**GDN prefill: the fork's TileLang kernel, or `gdn_tilelang_sm70` at the pin.**
The fork runs it with `--linear-attn-prefill-backend tilelang`. That
implementation was removed upstream before v0.5.22 —
`srt/layers/attention/linear/kernels/` ships only `gdn_cutedsl`,
`gdn_flashinfer` and `gdn_triton`, and `LINEAR_ATTN_KERNEL_BACKEND_CHOICES` has
no `tilelang` entry.

The flag in the block above therefore stays `triton`: it is the only value the
pin accepts, and the component is what puts the TileLang kernel back, by
answering `TritonGDNKernel.extend` on SM70. Add `gdn_tilelang_sm70` to
`SGLANG_RESTOMOD_COMPONENTS`, the same variable "Non-greedy requests" uses
below. Without it the launch is still valid and silent, and prefill runs on the
Triton chunk kernels.

**`fp8_moe` changes nothing on this launch.** The component exists to put the
experts on the TurboMind FP8 MoE GEMM, but with
`SGLANG_SM70_FP8_BACKEND=turbomind` the engine already routes them there. A
20-step decode trace is identical with and without it: 55 f16 cutlass launches,
4,140 and 1,380 TurboMind GEMM launches, 680 `main_kernel` launches, 906.6 ms
of kernel time against 911.7 ms.

Four reps each at 8k, 512 output tokens: 64.3 / 60.8 / 66.1 / 65.6 without the
component and 61.1 / 66.8 / 67.7 / 65.8 with it. Four requests aggregate 186.1
against 192.8, and the two ranges overlap. An earlier revision of this recipe
claimed 9-16% from the component; that was the spread between two two-rep
samples, not the component. Add it only if the launch does not already set
`SGLANG_SM70_FP8_BACKEND=turbomind`.

Receipt `evidence-ad0addd-recipes` records the effect at this commit: prefill
1,929.7 / 1,925.9 tok/s at 8,041 prompt tokens and 1,657.5 / 1,652.9 at 31,158,
against 684.6 on the Triton chunk kernels. The 63.9% attribution for that
baseline, and the kernel-level 3.2 ms against 75.2 ms, are in the lab note
`notes/gdn-prefill-backends.md` and in issue #80.

A 128-output-token A/B, four reps per arm and **unsigned**, is recorded in
`sglang_restomod/linear_attn_sm70/PROVENANCE.md`: prefill 1,648.4 / 1,645.1 /
1,639.0 / 1,637.4 against 684.7 / 684.2 / 684.4 / 684.3, and decode ranges that
overlap (TileLang 41.75-48.14, Triton 38.09-49.21), so no decode difference is
attributable to the component.

**`fp8_lm_head` buys about 7% at 8k.** The checkpoint keeps `lm_head` in
`ignored_layers`, so the head stays f16: 1.27 GB per card per call at TP2, five
calls a step (four draft, one verify; the draft shares the target head). The
component quantizes the head to 128x128 block FP8 after load and routes it to
the same TurboMind kernel `fp8_dense` uses, at half the weight bytes. It is
W8A16, so only the weights move.

Two A/B rounds on one host, one component toggled, 8k prompt, 512 output
tokens, one warmup and fresh reps:

| round | without | with | delta |
|---|---|---|---|
| 1 (3 reps) | 63.45 / 62.28 / 63.50, mean 63.08 | 71.08 / 69.90 / 70.12, mean 70.37 | +11.6% |
| 2 (4 reps) | 72.82 / 63.19 / 62.01 / 69.17, mean 66.80 | 68.35 / 68.14 / 64.28 / 75.09, mean 68.97 | +3.2% |
| pooled (7 reps) | mean 65.20 | mean 69.57 | +6.7% |

The accept length is 3.2-3.4 in every arm, so acceptance does not explain any
row. The run-to-run spread on this host is the same order as the effect: read
+6.7% as the estimate and +11.6% as the top of the measured range.

At 32k the two arms of round 1 read 56.08 / 53.89 / 50.92 (mean 53.63) against
54.63 / 54.31 / 54.39 (mean 54.44), inside the no-head spread. Why the head does
not move 32k is unconfirmed. The likely reason is that it leaves the critical
path there: the TurboMind GEMM total is flat at 19.0 ms from 8k to 32k while the
attention kernels grow 3.8x.

Quality, same two launches, one greedy prompt: the top-8 tokens keep their order
and the largest shift is 0.12 nats, against the 0.0383-0.0627 that the e5m2 KV
cache already costs. Two short factual answers are byte-identical.

Add `fp8_lm_head` after `fp8_dense` in `SGLANG_RESTOMOD_COMPONENTS`. It reuses
that component's patched `process_weights_after_loading` and `apply`; without
`fp8_dense`, or without the TurboMind FP8 kernel, it refuses to install and logs
a warning, so the launch stays on the f16 head.

Receipt `evidence-e3b28bb-fp8-lm-head` records the serving ladder of that launch:
8k 82.84 / 75.96 at one request, 120.26 / 122.89 at two and 212.12 / 200.68 at
four; 32k 56.73 / 57.82, 95.71 / 97.68 and 135.40 / 131.73.

## Environment

Complete list, as the launcher sets them. The Mamba dtypes are not optional:
the GDN layers fail or drift without them.

```
TORCH_CUDA_ARCH_LIST=7.0
FLASHINFER_DISABLE_VERSION_CHECK=1
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
SGLANG_MAMBA_CONV_DTYPE=float16
SGLANG_MAMBA_SSM_DTYPE=float16
SGLANG_SM70_FORCE_FP16=1
SGLANG_SM70_DENSE_GEMV=1
SGLANG_SM70_QWEN_FUSIONS=1
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
SGLANG_NUMA_BIND_V2=0
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```

The launcher also passes `--reasoning-parser auto --tool-call-parser auto
--sleep-on-idle`, which are serving preferences rather than performance flags.

## Measured

**Superseded for anything above ~12k by the table in "Pick the spec setting by
context".** These rows were taken before the draft-row fix, when the
draft-extend kernel still re-read the prefix per draft row; the MTP-on column
collapses from 12.7k up because of that kernel, not because of MTP. Kept for
history and for the short-context figures.

Decode tok/s, graph on, same harness for every row:

| Context | MTP off | MTP on | Accept len |
|---|---|---|---|
| 380 tok | 34.70 | **67.6-78.8** | 2.61-3.05 |
| 2.7k tok | 33.67 | **66.9-74.5** | 3.12-3.46 |
| 8.2k tok | 34.4 | **45.7** | 3.05 |
| 12.7k tok | 33.5 | 34.8 | 2.91 |
| 17.0k tok | **32.8** | 30.7 | 3.05 |
| 25.6k tok | **31.4** | 23.3 | 3.05 |
| 32.4k tok | **30.4** | 20.5 | 3.20 |

Prefill, warm process: 1,680-1,708 tok/s at 2k, 1,331-1,364 at 8k.

## Pick the spec setting by context

> **Pin status, 2026-11-02.** The table below is a fork measurement, taken with
> `draft-extend-native.patch` and `SGLANG_V100_DRAFT_NATIVE=1`. At the pin the
> equivalent is `SGLANG_V100_DRAFT_EXTEND_NATIVE=1` with `spec_sm70`, described
> above, and it changes the picture: with three draft steps and 40 KV splits the
> 27B measures **53.37 tok/s at 31.1k against 35.44 with MTP off**, at an accept
> length of 2.88-3.50. MTP therefore still wins at 32k on the pin, and this
> crossover table should not be used to choose the setting there. Nothing above
> 32k has been measured on the pin.

**MTP wins below ~48k and loses above it.** The draft-row fix
(`draft-extend-native.patch`, `SGLANG_V100_DRAFT_NATIVE=1`) removed the old
collapse, but it did not remove the per-row prefix read: every verify row and
every draft row still streams the whole prefix, so the step cost keeps growing
with context while the tokens per step stay at ~2.9.

Measured 2026-10-27 on lambda4 and lambda12, 2x V100, e5m2, graph on, greedy,
64 output tokens, 2 reps. The 128k arm was run on both hosts and agrees to
0.1%:

| context | MTP decode tok/s | no-MTP decode tok/s | MTP prefill | no-MTP prefill |
|---|---:|---:|---:|---:|
| 32k | 46.9 / 47.5 | 30.4 (fp16 arm) | 1,447-1,466 | - |
| 48k | 37.06 / 34.75 | 31.44 / 31.01 | 1,299 / 1,291 | 1,570 / 1,566 |
| 64k | 30.4 / 30.4 | 30.53 / 30.13 | 1,135 / 1,133 | 1,437 / 1,436 |
| 128k | 17.54 / 17.52 | 26.57 / 26.58 | 791 / 789 | 1,088 / 1,085 |
| 256k | - | 9.51 / 9.51 | - | 482 / 482 |

MTP also costs 17-27% of the prefill from 48k up, because the draft model
processes the same prompt tokens.

```
# <= 49152: MTP on; above: off. The launcher knob does this for you.
SGLANG_27B_SPEC=auto
```

If you set the flags by hand, keep `--speculative-*` off above 49152 and drop
`SGLANG_V100_DRAFT_NATIVE` with them. A purely decode-bound job at 48-64k can
still run MTP for a ~0-15% decode gain, at the prefill cost above.

## Long context

256k fits on 2x V100-32GB with a 1-byte KV cache: the pool holds 395,600
tokens, so the old 32,768 ceiling was the launch flag, not memory. A 1-byte
cache is not optional at the top end — the fp16 pool is 198,816 tokens and
cannot hold a 262,144-token request. `fp8_e4m3` and `fp8_e5m2` are the same
size; the rows below were measured with e5m2, before e4m3 was the
recommendation.

Measurements before 2026-10-28 (the rows at 64k and above carry MTP's cost
signature — see the crossover section — so read them as MTP rows):

| context | prefill tok/s | TTFT | decode tok/s |
|---|---:|---:|---:|
| 1k | 1,811 | 0.55 s | 77.1 / 85.3 (MTP) |
| 8k | 1,866 | 4.30 s | 56.5 / 81.3 (MTP) |
| 32k | 1,447-1,466 | 21.9 s | 46.9 / 47.5 (MTP) |
| 64k | 1,437 | 43.7 s | 30.5 (MTP) |
| 128k | 1,088 | 116-117 s | 26.6 (MTP) |
| 256k | 482 | 530 s | 9.51 (MTP) |

### Current: graph on, no MTP, two-CTA decode partial

27B-FP8, TP2, e5m2, greedy, 192 output tokens, prefix cache hitting after the
first rep, lambda12, `SGLANG_27B_SPEC=auto` (so no MTP above 49152) and the
decode CUDA graph on, which was only the launcher default from 2026-10-28:

| context | prefill tok/s | TTFT | decode tok/s |
|---|---:|---:|---:|
| 4k | 2,226 | 1.84 s | 34.26 |
| 32k | 2,963 | 11.1 s | 33.55 |
| 128k | 2,218 | 59.1 s | 30.83 |
| 256k | 2,021 | 129.7 s | **27.88** |

Against the table above: 256k decode **2.9x** and prefill **4.2x**; 128k decode
1.16x and prefill 2.0x. Only one of those three changes is a kernel (the
two-CTA decode partial, -28% per layer); the rest is turning off MTP at long
context and turning the graph on.

Decode is now flat-ish with context: 34.26 at 4k against 27.88 at 256k is a 19%
fall over a 64x increase, where the MTP ladder fell 66% from 8k to 256k.

Prefill is the weak term now, and it still falls with context (2,963 at 32k to
2,021 at 256k). That is the gather-and-recompute path rather than attention,
and it is the next thing worth attacking.

## Serving throughput

The rows above are one request at a time. A server is judged on several.
27B-FP8 TP2, e5m2 KV, MTP 5/6, 40 KV splits, 512 output tokens, two reps after
one warmup, sampling at the bench default of temperature 1. The single-stream
row prefills cold. The concurrent rows share one prefix, so every request is
decoding when the rate is taken.

The rate is the all-active window, latest first token to earliest last token,
over the `completion_tokens` the server reports. The burst column is the same
run over the earliest-first to latest-last window; it carries the stagger of
the later requests' prefill and is what a cold batch costs end to end.

Prompts are 8,034-8,041 and 31,359-31,385 tokens after the chat template. The
rows are the `bench.log` of the signed bundle `evidence-65c99d5-fp8-moe`, run
with 1024-token chunks.

| prompt | concurrency | aggregate tok/s | per request | burst tok/s |
|---|---:|---:|---:|---:|
| 8k | 1 | 63.0 / 66.6 | 63.0 / 66.6 | same |
| 8k | 2 | 120.5 / 110.2 | 60.2 / 55.1 | 117.6 / 107.7 |
| 8k | 4 | 195.6 / 187.7 | 48.9 / 46.9 | 187.6 / 180.7 |
| 32k | 1 | 54.8 / 55.0 | 54.8 / 55.0 | same |
| 32k | 2 | 86.4 / 84.4 | 43.2 / 42.2 | 83.9 / 81.9 |
| 32k | 4 | 114.2 / 120.3 | 28.5 / 30.1 | 107.1 / 112.1 |

At 8k batching is close to linear: 63 to 120 to 196 tok/s, 1.9x and 3.1x the
single stream. At 32k it flattens to 1.6x and 2.1x, because every step reads
the whole prefix once per row. Per-request rate falls in both cases, so
concurrency buys throughput at the cost of latency.

The two reps agree within 1% at 8k and within 7% at 32k.

Earlier revisions of this table quoted a 128-token run over the burst window,
which read four 8k requests at about 108 tok/s. That window is mostly prefill
stagger at 128 tokens, and the draft needs the longer decode to reach steady
state.

`scripts/gpu_suite.sh --concurrency 1,2,4` reproduces the ladder in one bundle.

### Prefill chunks

A decode batch freezes for about one prefill chunk when a new request arrives.
Three requests decoding 8k prompts, then a fourth arrives with an 8k prompt:

| `--chunked-prefill-size` | newcomer prefill | decoder gap during it |
|---:|---:|---:|
| 4096 (default) | 4.26 s | 1,032 ms median, 1,990 ms max |
| 1024 | 4.41 s | 289 ms median, 507 ms max |
| 512 | 4.84 s | 277 ms median, 277 ms max |

The steady decode gap is 73 ms at every setting. 1024 is the knee: 3.6x less
stall for 3% more prefill time, while 512 buys 4% more stall reduction for 10%
more prefill. Prefill throughput barely moves across the three, so pick the
chunk for the latency budget.

Longer context moves the ceiling. On the 256k launch (`--context-length
262144`, mem 0.82, same MTP and KV), 512 output tokens, one rep:

| prompt tokens | concurrency | decode tok/s | prefill tok/s | TTFT |
|---:|---:|---:|---:|---:|
| 64,263 | 1 | 41.6 | 1,321 | 48.6 s |
| 64,263 | 2 | 56.7 aggregate | same | same |
| 128,564 | 1 | 29.7 | 1,017 | 126.5 s |

Long-context concurrency is bounded by the KV pool, not by the scheduler. Four
64k requests are 256k KV tokens, and that run returned no tokens: the server
process died. Two at 64k fit, and one at 128k fits. Size long-context
concurrency from the pool, not from the 32k ladder.

## Host KV tier

`SGLANG_27B_HICACHE_GB=N` (`--enable-hierarchical-cache --hicache-size N`) is
worth turning on whenever the working set does not fit the GPU pool. 16
distinct 30,720-token documents (492k tokens) against the 397k-token pool:

| arm | round-2 TTFT |
|---|---|
| off | 17.3 s (16/16 re-prefill) |
| on, 32 GB/rank | **0.20 s (16/16 hit)** |

The tier is 1,029,904 tokens (16.87 GB) of host KV per rank plus 15.14 GB of
host Mamba state, 2.6x the device pool.

**Decode after a restore.** The copy costs TTFT, not steady state. Measured
2026-11-03 on lambda12, same launch, 14 distinct 30k-token documents against a
30k prompt, greedy: the restored prefix answers its first token in 0.43 / 0.33 s
and decodes at 43.84 / 42.77 tok/s, against 43.25 / 49.36 with a cold prefill.
The spread between two cold reps is as wide as the gap, so a restored prefix
decodes like a prefilled one; the difference is the ~480 MB per rank that
crosses PCIe before the first token.

**Sizing.** `--hicache-size` is per rank, and SGLang cannot bound it
automatically in this container (`Cannot determine cgroup memory headroom`).
Size it as (working set - GPU pool) / ranks, and keep the total under an eighth
of host RAM: 503 GB here, so 32 GB per rank is 13%.

### The disk tier: `write_back` plus `wait_complete`

`--hicache-storage-backend file` needs the native hash extension, which
JIT-builds against OpenSSL. The CUDA image ships only the runtime
(`libcrypto.so.3`), so the build fails out of the box;
`patches/sglang-sxm2/hicache-openssl-fallback.patch` in the lab repo takes the
headers and the library from the interpreter's conda prefix and links by path,
so the image's runtime library satisfies the recorded `NEEDED`. No `libssl-dev`
is required.

Two settings decide whether the disk tier does anything at all. 24 distinct
30,720-token documents (737k tokens) against a 397k-token device pool and an
**8 GB** host tier (8.4 documents), two passes in order:

| arm | round 1 | round 2 |
|---|---|---|
| `timeout` (default), `write_through` | 17.2-17.7 s | 17.5-18.3 s |
| `wait_complete`, `write_through` | 4.0-4.5 s (docs 0-7), 17.6-19.6 s (docs 8-23) | 17.5-17.8 s |
| `wait_complete`, **`write_back`** | 15/24 at 1.0-1.1 s, 4 at ~9.5 s, 5 at 17.8 s | **24/24 at 1.0-1.1 s** |

- `--hicache-storage-prefetch-policy wait_complete`. The default `timeout` gives
  up after 1.0 s + 0.25 ms per page, and the hybrid all-or-nothing check then
  discards the partial read because the Mamba pool cannot cover it: all 36
  prefetches were discarded and nothing hit.
- `--hicache-write-policy write_back`. Under `write_through` the prefetch asks
  for the whole document but `evict_host()` frees nothing, so it falls back to
  the ~2,900 free tokens and that is discarded too. Under `write_back` the
  allocation succeeds outright and the pool reclaims for the prefetch.

That combination serves a working set **twice the host tier** from disk at 1.0-1.1 s
TTFT against 17.3 s to re-prefill: 16x. Replicated with 16 documents on a host
whose page cache holds the 40 GB corpus: 16/16 at 0.19-0.62 s. A restored prefix
also decodes at the same rate as a freshly prefilled one (median 32.63 vs
32.58 tok/s at 32k, no MTP, over 16 documents) and greedy output is
byte-identical, so the win is not paid for during generation and does not change
what the model says.

## KV dtype

> **Pin status, 2026-11-02.** `--kv-cache-dtype fp8_e4m3` does not start on sm70
> at this pin. The grouped decode kernel
> (`sglang/kernels/ops/attention/decode_attention.py:853`,
> `_decode_grouped_att_m_fwd`) compiles for `fp8e4nv`, and Triton refuses:
> `type fp8e4nv not supported in this architecture. The supported fp8 dtypes are
> ('fp8e4b15', 'fp8e5')`. The server dies before the model loads, so no
> attention backend choice avoids it.
>
> `fp8_e5m2` does start, and until #129 it cost **1.38x decode** against fp16 at
> 32k, because the fp8 clauses in `attention_v100` sent the draft extend to the
> Triton extend kernel (31.2 ms per launch, 32 launches) instead of the native
> grouped path. #129 defers to the draft-extend gate in all four dispatch sites,
> and with `SGLANG_V100_DRAFT_EXTEND_NATIVE=1` on, as above, the cost is gone:
>
> | arm, 32k, greedy | decode tok/s, four 128-token reps | mean |
> |---|---|---:|
> | fp16 KV | 53.32 / 56.00 / 49.75 / 54.64 | 53.43 |
> | e5m2 KV, before #129 | 38.20 / 34.93 / 39.31 / 33.32 | 36.44 |
> | e5m2 KV, after #129 | 61.69 / 57.37 / 58.83 / 57.33 | 58.81 |
>
> The signed bundle for #129, at temperature 1, agrees at 32k: 54.39 / 46.69
> against the fp16 40-split rows' 54.42 / 47.13, and 67.11 / 65.54 at 8k against
> 73.33 / 66.43. So the 1-byte cache is now a capacity win at 32k with a 5-10%
> cost at 8k, not a 1.4x decode penalty. It still passes a 262k request on the
> same cards that fp16 cannot.
>
> **fp16 KV stays the default** because it is exact and faster at 8k; pass
> `fp8_e5m2` when the context needs the capacity. The comparison further down
> was measured on the fork and stands as provenance, not as guidance.

`fp8_e4m3` is the recommendation where the decode kernel supports it. It is the
same size as e5m2, runs at the same
speed in the SM70 decode kernel, and moves the model's own next-token
distribution less.

The 1-byte formats first: #51 was two uninitialised reads in the decode partial,
not a numerics problem; #79 put the e5m2 EAGLE target verify on the native
grouped decoder, and at 32k with MTP it measures 47.4 tok/s against fp16's
49.9 / 41.6 under the same sampling, with ~2x the prefill (1,447 vs 704). The
"per-row decoder needs fp16 KV" rule is gone.

e4m3 against e5m2, 27B-FP8 TP2, four fixed continuations, per-token
log-probability against an fp16 cache (the lab's `notes/kv-dtype-quality.md`):

| | e5m2 | e4m3 |
|---|---:|---:|
| mean abs log-prob shift, per token | 0.0627 | **0.0383** |
| rms shift | 0.1030 | **0.0711** |
| worst token | 0.5143 | **0.4350** |
| closer to fp16 on | 60/177 | **117/177** (p=2.2e-5) |
| signed shift, whole set | +0.29 | -0.99 |

Cost: none measurable. Served at 122,880 tokens, 30.95 against 30.93 tok/s;
per attention layer at 256k, 0.635 against 0.629 ms. Range: the kernel runs
with `k_scale=1.0`, so e4m3's 448 ceiling matters — the measured K maximum on
this model is 24.97, an 18x margin. Capacity: both are 1 byte, and the pool
holds 392,560 tokens against fp16's 196,272.

The e4m3 arm is slightly *less* confident overall: its signed shift is negative
on all four prompts, -0.0056 nats per token, about 0.6% of a perplexity point.
e5m2 has no such bias but a 1.6x larger per-token error, which costs more in
expectation. Neither magnitude reorders a greedy decode.

## Notes

- The checkpoint carries its own MTP head (`mtp_num_hidden_layers=1`), so no
  draft model is needed. There is no QSA ring here, so the draft chain is not
  capped at 4 tokens the way it is on Flash-Next.
- Decode graph capture needs `graph-metadata.patch`; without it the graph runner
  calls `init_forward_metadata_out_graph`, the base class leaves it a no-op, and
  capture dies on `swa_out_cache_loc`.
- The first request after startup is much slower while prefill kernels JIT. A
  benchmark has to warm up or it quotes the warmup.

Run records: [docs/runs/2026-10-26-qwen3.8-27b-mtp-lambda12.md](../docs/runs/2026-10-26-qwen3.8-27b-mtp-lambda12.md),
[docs/runs/2026-10-26-qwen3.8-27b-kv-dtype-mtp-crossover.md](../docs/runs/2026-10-26-qwen3.8-27b-kv-dtype-mtp-crossover.md).
Launcher: `scripts/serve_sglang_27b_v100.sh` in the lab repository.
