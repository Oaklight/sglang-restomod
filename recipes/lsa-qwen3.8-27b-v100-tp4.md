# Qwen3.8-27B-FP8 on 4x V100-SXM2, TP4, LSA selection

| | |
|---|---|
| Model | `Qwen3.8-27B-FP8` (block FP8, hybrid GDN + full attention) |
| Hardware | 4x Tesla V100-SXM2-32GB (sm70) |
| Topology | TP4, no EP |
| Context | 32,768 (measured at 2,048 and 16,384 prompt tokens) |
| Engine | `sgl-project/sglang` `v0.5.22` @ `baa9befa` with this layer installed |
| Components | `fp8_dense`, `lsa_sm70` |
| Date | 2026-11-04 |

The full-attention layers read a Latent Sparse Attention selection: keys are
clustered in the released D-LSA latent space, each query scores bucket
centroids, and it reads the selected keys plus a 512-key window and 4 sinks.
The 48 GDN layers are untouched.

## Flags

```
--dtype float16 --quantization fp8
--attention-backend lsa_sm70
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
--page-size 1 --disable-cuda-graph
--tp-size 4 --context-length 32768 --mem-fraction-static 0.80
--max-running-requests 4 --chunked-prefill-size 1024 --max-prefill-tokens 8192
--sampling-backend pytorch
```

`--sampling-backend pytorch` is required: the FlashInfer sampler refuses sm70,
so a non-greedy request kills the server without it.

`--disable-cuda-graph` is required: the selection is host-orchestrated and the
backend refuses capture at startup. There is no captured prefill or decode.

`--page-size 1` is required: an LSA list is a set of single tokens. The KV cache
must be bf16 or fp16; an fp8 cache is refused at startup. Plain tensor parallel
only: data-parallel attention, context parallel and speculative decoding are
refused.

## Install

`lsa_sm70` is a registration shim; the implementation is the standalone
package, pinned in `UPSTREAM_PIN.toml`:

```sh
pip install 'lsa-sm70 @ git+https://github.com/Oaklight/lsa-sm70@3a4798093224933988175ee685084d521474da0b'
```

## Environment

```
SGLANG_RESTOMOD_COMPONENTS=fp8_dense,lsa_sm70
SGLANG_RESTOMOD_LSA_MAP=/path/to/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt
SGLANG_RESTOMOD_LSA_TOP_K=2048 SGLANG_RESTOMOD_LSA_TOP_P=512
SGLANG_RESTOMOD_LSA_WINDOW=512 SGLANG_RESTOMOD_LSA_SINK=4
SM70_TURBOMIND_SO=/path/to/_sm70_turbomind_v100.so
SGLANG_SM70_TURBOMIND_OPS_PATH=/path/to/_sm70_turbomind_v100.so
SGLANG_SM70_FORCE_FP16=1 SGLANG_SM70_DENSE_GEMV=1 SGLANG_SM70_QWEN_FUSIONS=1
SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage FLASHINFER_DISABLE_VERSION_CHECK=1
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=7.0
```

The map is the released D-LSA projection file for Qwen3.8-27B, 16 layers
(3, 7, ..., 63) of `[768, 64]` `W_Q_idx`/`W_K_idx`.

## Accuracy against the dense backend

The same prompt ids, greedy, 32 output tokens, against `--attention-backend
tilelang_fa_v100` on the same launch:

| prompt tokens | output tokens identical | mean abs log-prob shift | max shift |
|---|---:|---:|---:|
| 2,048 | 32/32 | 6.7e-4 nats | 3.7e-3 |
| 16,384 | 32/32 | 1.7e-4 nats | 1.5e-3 |

## Speed

Bench at concurrency 1, 32 output tokens, one warmup and two reps, from the
evidence bundle:

| prompt tokens | TTFT | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| 2,018-2,020 | 4.53 s | 445 | 7.68-7.70 |
| 16,069-16,074 | 61.9 s | 259.5-259.7 | 7.97-8.01 |

These are not tuned numbers: the selection runs on the host-side path and there
is no CUDA graph. The point of this recipe is that the model answers correctly
with a subquadratic read.

## Receipt

The evidence bundle for the commit that carries this recipe is signed with the
V100 operator key and posted on PR #182; the `gpu-receipt` check carries the
bundle digest. It records `gpu_tests.txt` (13 passed / 1 skipped) and the launch
check `attention_backend=lsa_sm70`.
