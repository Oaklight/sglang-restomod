# Qwen3.8-27B-FP8 on 2x V100-SXM2, TP2, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-27B-FP8` (block FP8, hybrid dense + GDN) |
| Hardware | 2x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP2, no EP |
| Context | 32,768 to 262,144 (measured at both) |
| Engine | Two stacks, and the text says which rows are which. The fork tables were measured on `dg1kjd/sglang-sxm2` @ `73abfe1` plus the lab patch set (`patches/sglang-sxm2/` in `Oaklight/lambda-v100-lab`). Every figure marked "at the pin", and every component row, was measured on `sgl-project/sglang` `v0.5.22` @ `baa9befa` with this layer installed. |
| Components | `attention_v100`, `fp8_dense`, `spec_sm70`, `gdn_tilelang_sm70` |
| Date | 2026-11-02 (KV dtype: e4m3 at the fork, fp16 at the pin — see "KV dtype") |

## Flags

```
--dtype float16 --quantization fp8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
# --kv-cache-dtype fp8_e5m2           # optional 1-byte cache for 256k; costs
                                      # 1.38x decode at the pin. See "KV dtype".
--tp-size 2 --context-length 262144   # max_position_embeddings
--mem-fraction-static 0.85 --max-running-requests 4
--chunked-prefill-size 4096 --max-prefill-tokens 8192
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
> `fp8_e5m2` does start, and at 32k it costs **1.38x decode** against fp16 for
> the same prefill: 29.02 / 32.24 / 30.92 / 34.55 tok/s against 41.52 / 48.11 /
> 41.62 / 43.66, four 128-output-token reps, one server restart apart. On this
> launch it allocated a 305,280-token pool.
>
> So **fp16 KV is the pin default**, and the 1-byte cache is a capacity trade for
> 256k rather than a free win. The comparison below was measured on the fork and
> stands as the reason to keep the format on the roadmap, not as a reason to
> pass the flag today.

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
