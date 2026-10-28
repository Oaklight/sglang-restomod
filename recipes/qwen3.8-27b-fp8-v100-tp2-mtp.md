# Qwen3.8-27B-FP8 on 2x V100-SXM2, TP2, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-27B-FP8` (block FP8, hybrid dense + GDN) |
| Hardware | 2x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP2, no EP |
| Context | 32,768 to 262,144 (measured at both) |
| Engine | `dg1kjd/sglang-sxm2` @ `73abfe1` + the ten lab patches (`patches/sglang-sxm2/`) |
| Components | none (the dense path needs no restomod component) |
| Date | 2026-10-28 (KV dtype: e4m3; long context, HiCache) |

## Flags

```
--dtype float16 --quantization fp8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend tilelang --linear-attn-decode-backend triton
--kv-cache-dtype fp8_e4m3             # 1 byte; required for 256k. See "KV dtype".
--tp-size 2 --context-length 262144   # max_position_embeddings
--mem-fraction-static 0.85 --max-running-requests 4
--chunked-prefill-size 4096 --max-prefill-tokens 8192
--disable-prefill-cuda-graph          # decode graph stays on
# MTP only up to ~48k: see "Pick the spec setting by context" below.
# SGLANG_27B_SPEC=auto in the launcher does exactly this.
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

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

`fp8_e4m3` is the recommendation. It is the same size as e5m2, runs at the same
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
