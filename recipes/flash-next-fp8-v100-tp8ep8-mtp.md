# Qwen3.8-Flash-Next-FP8 on 8x V100-SXM2, TP8 + EP8, CUDA graph + MTP

| | |
|---|---|
| Model | `Qwen3.8-Flash-Next-FP8` (512 experts, block FP8 `[128,128]`, QSA + GDN) |
| Hardware | 8x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP8 + EP8 (64 local experts per rank) |
| Context | 8,192 |
| Engine | `dg1kjd/sglang-sxm2` @ `73abfe1` + `sm70-fixes.patch` + `graph-metadata.patch` + `qsa-decode-h3.patch` |
| Components | `fp8_moe` |
| Date | 2026-10-26 |

## Flags

```
--dtype float16 --quantization fp8
--tp-size 8 --ep-size 8
--attention-backend tilelang_fa_v100
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
--kv-cache-dtype auto --context-length 8192
--mem-fraction-static 0.80 --max-running-requests 1
--chunked-prefill-size 4096 --max-prefill-tokens 8192
--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**GDN prefill runs on Triton at the pin.** The fork used
`--linear-attn-prefill-backend tilelang`. That implementation was removed
upstream before v0.5.22 — `srt/layers/attention/linear/kernels/` now ships only
`gdn_cutedsl`, `gdn_flashinfer` and `gdn_triton`, and
`LINEAR_ATTN_KERNEL_BACKEND_CHOICES` has no `tilelang` entry. Triton is the only
path that exists at the pin, so any prefill number measured on the fork is not
directly comparable to one measured here.

## Environment

Complete list, as the launcher sets them. The Mamba dtypes are not optional:
the GDN layers fail or drift without them.

```
SGLANG_PLUGINS=restomod
SGLANG_RESTOMOD_COMPONENTS=fp8_moe
TORCH_CUDA_ARCH_LIST=7.0
FLASHINFER_DISABLE_VERSION_CHECK=1
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
SGLANG_MAMBA_CONV_DTYPE=float16
SGLANG_MAMBA_SSM_DTYPE=float16
SGLANG_SM70_FORCE_FP16=1
SGLANG_SM70_DENSE_GEMV=1
SGLANG_SM70_QWEN_FUSIONS=1
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```

Audited against the pin's `srt/environ.py`: **`SGLANG_MAMBA_CONV_DTYPE` and
`SGLANG_MAMBA_SSM_DTYPE` are the two that matter** — they exist and are what
keep the GDN state in fp16. `SGLANG_SM70_FORCE_FP16`, `SGLANG_SM70_DENSE_GEMV`,
`SGLANG_SM70_QWEN_FUSIONS` and `SGLANG_CUSTOM_ALLREDUCE_ALGO` are **not read
anywhere at the pin** (fork-only switches); they are inert there and kept only
so this recipe still describes the fork launcher.
`SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS` is live — the component reads it.

`SGLANG_CUSTOM_ALLREDUCE_ALGO` also accepts `2stage`; measured at 87.9 vs 86.5
tok/s on a short prompt and 80.7 vs 84.0 at 4.8k, i.e. no difference.

## Measured

| Case | Prompt | Decode | Accept len |
|---|---|---|---|
| explain | 37 tok | **82.9-86.5 tok/s** | 2.78-2.88 |
| memo | 45 tok | **67.9 tok/s** | 2.33 |
| long | 4,856 tok | **83.2 tok/s** | 2.69 |

Against the same graphed configuration without MTP (30.8-36.9 short, 25.96-36.65
at 3k) that is 2.2-2.7x. Cold 4,856-token prefill: 0.96 s, about 5,061 tok/s.

## Notes

- **EP is required.** block-128 experts cannot be TP-sharded (640/8 = 80), so
  TP8 alone leaves each rank with a partial expert.
- **The draft chain is capped at 4 tokens.** `--speculative-num-draft-tokens 6`
  fails at startup: `Qwen QSA requires speculative_num_draft_tokens <= the QSA
  compress ratio (4): the pending index-key ring holds one group`. This is
  structural, not a tuning choice.
- QSA decode needs `qsa-decode-h3.patch`, which templates the SM70 kernel over
  the query-head group so the 3-head layout at TP8 works.
- The first request after startup takes about 85 s (JIT and warmup after graph
  capture). Benchmark with a warmup request.
- Under MTP the all-reduce becomes the top cost: 12.2 of 29.1 ms per step, in
  305 collectives above 500 us. They never overlap compute. See #64.

Launcher and full measurement note: `benchmarks/flash-next-fp8-sglang-lambda4-tp8.md`
in the lab repository.

## Measured at the pin (restomod, 2026-10-31)

Same hardware, same model, `sglang-restomod` on upstream `v0.5.22 @ baa9befa`
with no patches. Launcher: `pin_launch_fn_mtp.sh` on lambda1.

```
--dtype float16 --quantization fp8
--tp-size 8 --ep-size 8 --page-size 64
--attention-backend qsa_v100
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
--context-length 4096 --mem-fraction-static 0.85 --max-running-requests 1
--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

| Case | Decode | Accept len |
|---|---|---|
| no MTP, 76-token prompt, 128 out | **40.0 tok/s** | — |
| MTP3/4 draft, same prompt | **73.9 tok/s** | 2.83-2.85 |
| MTP3/4 draft, temperature 0, three runs | **98.3—106.2 tok/s** | 3.48—3.70 |

Per-step time on the greedy row is `accept / throughput`, i.e. 34.8—35.4 ms.
That is the comparable quantity against the NVFP4 four-card profile in
`recipes/flash-next-nvfp4-v100-tp4-mtp.md`; the 73.9 tok/s row above is the
default-sampling run and is not comparable to it.

Without MTP the pin is 2x behind the fork's 30.8-36.9 measured on the same box;
with MTP it is 12-15% behind the fork's 82.9-86.5. Closing that gap needs the
collectives, not the kernels: `ncclDevKernel_AllReduce_Sum_f16_RING_LL` was 24.5%
of CUDA time (about 101 all-reduces per step, 79.5 us each) in the profile
recorded on issue #94.

`--attention-backend qsa_v100` is required. The registry does not reach the
full-attention backend for this model: `attention_registry.py` builds its own
`QwenSparseAttnBackend` for hybrid GDN + QSA models, and that class needs
flash_attn for its packed varlen fallback, which sm70 cannot run. The component
rebinds the class at plugin load (`backend_registry.install_qsa_swap`).

### Card-count floor

The FP8 checkpoint is about 120 GB. On 32 GB cards:

| Cards | Result |
|---|---|
| 2 (TP2/EP2) | load-time OOM, 30.95 GiB allocated per card |
| 4 (TP4/EP4) | load-time OOM, 30.68 GiB allocated per card |
| 8 (TP8/EP8) | runs |

So Flash-Next needs all 8 cards on this hardware; the 2- and 4-card runs belong
to the 27B model.

### Topology matters

lambda1's eight V100s sit in two NVLink islands (0-3 and 4-7, `SYS` between),
so the eight-rank group is not full NVLink. SGLang says so at startup:
`CustomAllreduce is disabled because it's not supported on more than two
PCIe-only GPUs`. Every all-reduce therefore crosses the socket boundary, which
is the cost the pin still pays relative to the fork.
