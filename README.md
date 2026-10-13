# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) adapted for **eight V100 SXM2 32 GB GPUs
in two NVLink quads**. We rebuilt useful SM70 support from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
effectively rebasing it into an independent fork. Qwen3.8-Flash-Next FP8 and
GLM-5.3-Flash NVFP4 work is consolidated on `main`.

`main` contains validated changes and stays frozen during the unified GLM
attention rewrite on `dev`. The [FA fork](https://github.com/heislera763/flash-attention-v100)
owns native CUDA sparse, vision and KDA attention; SGLang owns thin adapters,
metadata and cache/state lifetimes. Advance `main` after correctness and model
throughput checks. Qwen retains its existing backend during this work.
Clone remotes: `origin` is this fork; `upstream` is `sgl-project/sglang`.

Integrated upstream: [3831e7e0](https://github.com/sgl-project/sglang/commit/3831e7e0918052be50ef38b3ff62154573e213d7), pinned 2026-10-10.
Hardware adapters live in [v100_plus/sglang_v100_plus](v100_plus/sglang_v100_plus),
reusable operators in [python/sglang/kernels/ops](python/sglang/kernels/ops), and
build/Marlin changes in [v100_plus/aot](v100_plus/aot) and [v100_plus/patches](v100_plus/patches).
[Provenance](v100_plus/provenance.json) and [recorded core changes](v100_plus/core-patches.json)
make upstream updates auditable. Services, benchmark clients, traces and project notes
stay outside Git. This assumes familiarity with SGLang.

## Checkpoints

Pinned benchmark sources; sizes are weight-file totals, **not runtime VRAM**.

| Checkpoint | Revision | Composition | Files |
| --- | --- | --- | ---: |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13) | `236dfdf28582` | Official 128×128 block FP8, BF16 exclusions; FP8 PLE/MTP experts | 172.78 GiB |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) | `da920bb0b9f4` | ModelOpt abs-max NVFP4; main shared experts remain BF16; FP8 KV metadata | 190.40 GiB |
| Local GLM MTP FP8 | NVIDIA source above | Routed MTP experts converted to128×128 E4M3 block FP8/W8A16; other tensors unchanged. Used by current PP2 measurements; local `PROVENANCE.json` records conversion. | 7.10 GiB |

NVIDIA GLM is the active GLM checkpoint. It uses an abs-max recipe, with BF16
shared experts and MTP exclusions; publisher alone does not prove better quality.
TP8 ordinary/MTP and PP2 ordinary/MTP are the reference layouts. The PP2 MTP
module-load OOM is handled by cold-load allocator reclamation, without changing
checkpoint, arithmetic or workload limits. NVIDIA Qwen NVFP4 remains deferred;
full GLM FP8 routed weights alone need roughly 283.5 GiB.

At checkpoint-native context, cache estimates below exclude weights, state,
execution scratch and required page/sentinel/speculative overhead:

| Model | Native tokens | TP8 main KV + index / GPU | PP2 largest main KV + index / GPU |
| --- | ---: | ---: | ---: |
| Qwen FP8 weights, FP16 KV | 262,144 | 3.19 GiB | 1.59 GiB |
| NVIDIA GLM | 1,048,576 | 11.35 GiB | 6.19 GiB |

Qwen MTP adds about0.27GiB on its owning stage. Native FP16 PP2/MTP3
capacity262106input+32output and request reuse pass. GLM's FP16 latent KV is replicated
across TP ranks. The current GLM PP2/MTP3 session limit is **262,144** with a
262,656-slot pool (~1.55GiB/GPU including target/draft KV and index storage).
A 262,106-input/32-output sampled capacity probe and subsequent request reuse passed;
this is functional capacity coverage, not a long-context quality claim. Larger
MTP windows and native1M remain unqualified. The TP8 launch below is a short-context
reference; the PP2 variation gives the working session configuration.

## SM70 execution

Volta supports FP16 Tensor Cores. FP4/FP8/BF16 checkpoint storage is decoded or cast
in software; it does not imply native arithmetic in those formats.

| Operation | Implementation / contract |
| --- | --- |
| NVFP4 GEMMs | [marlin-v100](https://github.com/heislera763/marlin-v100) at `95b1e62e` plus local patches: W4A16, FP16 WMMA with FP32 accumulation; block/global scales remain distinct. Small routed groups use [Volta MMA](python/sglang/kernels/ops/gemm/sm70_nvfp4_routed.py) with M16 tiles and K-major decoded weights; FP16 dequantization/output and FP32 accumulation/scaling remain. Unused alternate-backend MoE scale buffers alias active scales, saving17.72GiB across GLM's8GPUs. |
| Block-FP8 GEMMs | Patched Marlin and small-row vector kernels: W8A16, complete 128×128 scale blocks. Qwen EP4/EP8 preserves whole experts. |
| FP16 routed experts | [Volta MMA GEMM](python/sglang/kernels/ops/moe/sm70_fp16.py), used by NVIDIA GLM's unquantized MTP experts. FP16 inputs/weights/output, FP32 accumulation and router multiply; upstream routing, activation and expert sum retained. Automatically selected by the SM70 plugin; unsupported epilogues fail under strict dispatch. |
| Qwen QSA/GDN/HC | Direct FP16-cache sparse prefill/decode/verification, masked Tensor Core prefill, SM70 GDN and gated-residual kernels; HC eager prefill partitions within each quad, including TP8. |
| GLM attention | [Canonical V100 CUDA backend](python/sglang/kernels/ops/attention/v100.py) from the [FA fork](https://github.com/heislera763/flash-attention-v100): selected-key FP16 sparse prefill/decode/MTP, FP32 reductions and FP16 probabilities/partials. Empty tiles skip identity work. The model-native FP8 indexer retains [live-history MMA scoring](python/sglang/kernels/ops/attention/dsa/sm70_indexer_decode.py) and bounded prefill scoring. |
| GLM KDA / mHC | Native CUDA full-K FP32 recurrence, FP16 state and per-step verification carry. Prefill prepares normalized Q/K and gates once in runner-owned FP32 working vectors. KDA remains scalar FP32 CUDA; Tensor Core chunk formulation is future work. mHC retains its existing SM70 projection/RMS/Sinkhorn paths. |
| Storage | Unquantized FP16 attention KV for both models; Qwen FP16 QSA index, GLM's separate model-native FP8 indexer. KV-cache quantization is excluded from the project recipes. FP16 recurrent state is retained. |
| Cold module loading | Return idle Torch allocator blocks to CUDA when first-use Triton loads have little headroom; leave live tensors unchanged. Skip capture/custom arenas; steady kernel replay has no callback. |
| Vision / DFlash2 | GLM uses the FA port’s native FP16 D64 varlen attention; Qwen retains SDPA vision. [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) FP16/local W8A16 have prior experimental coverage; NVIDIA-target qualification is pending. |

Strict dispatch checks backend selection; it does not imply every operation is
Volta-specific. Routing/activation/reduction preparation still uses general CUDA,
Torch and Triton kernels; GLM vision uses native FP16 FA and Qwen uses SDPA. Compact GLM FP8 MTP prefill uses
the generic SM70 Marlin scale path, while the explicit block-scale fast path is
currently qualified only for Qwen shapes.

For FP16 GLM, the plugin resolves checkpoint-inferred `auto` sparse KV to FP16,
including NVIDIA's FP8 KV metadata; weight quantization is unchanged. Explicit cache
choices remain explicit. FP32 storage does not remove FP16 internal arithmetic:
Qwen's current FP32-state path fails an existing operator bound; GLM's passes but
showed no speed gain. Strict dispatch rejects unsupported guarded paths; declared
Marlin/cuBLAS/Torch CUDA backends remain valid.

The `dev` attention profile uses one selector, `--attention-backend v100_plus`,
for sparse, KDA and vision. It requires FP16 compute/state, compute-dtype `auto`
KV, `no_buffer` recurrent caching and one running request. Conflicting backend
choices fail before loading. Legacy GLM sparse hooks and separate kernel modules
have been removed; Qwen keeps its existing operators.

## Build and reference launches

Python 3.12, `uv`, CUDA 12.9; [setup](v100_plus/setup.sh) builds pinned SM70 AOT/Marlin/canonical FA
kernels. Use a fresh shell and the common block below for each recipe. Device order
must follow your topology. Local directory names are conventions; for Hub loading,
use the checkpoint ID and full `--revision` above. These are manual single-request
references, not installed services.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
git remote add upstream https://github.com/sgl-project/sglang.git
bash v100_plus/setup.sh
export CUDA_HOME=/usr/local/cuda-12.9 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH" OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_PLUGINS=v100_plus SGLANG_V100_PLUS=1 SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm" SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
common=(
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531 --dtype float16 --trust-remote-code
  --moe-runner-backend marlin --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto
  --disable-overlap-schedule --disable-radix-cache
  --mamba-ssm-dtype float16 --max-running-requests 1 --disable-prefill-cuda-graph
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1
)
```

**Qwen FP8, TP4×PP2/EP4, ordinary** (`http://<host>:9000/v1`, key `test-only`):

Native QSA prefill/decode/verification read FP16 KV directly. For this pinned
checkpoint, `--dtype float16 --kv-cache-dtype auto` allocates FP16 KV; the
indexer and recurrent state also use FP16. Strict dispatch remains enabled.

```bash
export SGLANG_PP_LAYER_PARTITION=24,24 SGLANG_OPT_SM70_HC_PREFILL_SP=1
export SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" --served-model-name qwen3.8-flash-next \
  --quantization fp8 --json-model-override-args '{"language_model_only":true}' \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --context-length 262144 --max-total-tokens 262272 \
  --chunked-prefill-size 4352 --attention-backend triton --mm-attention-backend sdpa \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype auto --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

**GLM NVIDIA NVFP4, TP8/EP1, ordinary** (fresh shell plus common block):

```bash
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_FUSE_MHC_POST_PRE=0
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/nvidia-GLM-5.3-Flash-NVFP4" --served-model-name glm5.3-flash \
  --quantization modelopt_fp4 --fp4-gemm-backend marlin --tensor-parallel-size 8 --expert-parallel-size 1 \
  --context-length 12288 --max-total-tokens 12288 \
  --mem-fraction-static 0.92 --chunked-prefill-size 2048 --attention-backend v100_plus \
  --page-size 256 --kv-cache-dtype auto --mamba-radix-cache-strategy no_buffer \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":true}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

| Variation | Change from reference |
| --- | --- |
| Qwen TP8 | TP8/EP8; omit PP and unset `SGLANG_PP_LAYER_PARTITION`/metadata flag. Keep HC/chunk4352. |
| GLM PP2 MTP | TP4/PP2/EP1; `SGLANG_PP_LAYER_PARTITION=24,21`; context **262144**, total tokens **262656**, fraction **.95**, chunk2048. Add MTP flags below and `--speculative-draft-model-path "$HOME/models/sglang/local-GLM-5.3-Flash-MTP-FP8" --speculative-draft-model-quantization fp8` for the measured compact draft. Images are enabled; vision weights reside on the first PP stage. |
| MTP, either model | Add `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 --speculative-use-rejection-sampling`. PP2 also exports `SGLANG_ENABLE_PP_SPEC=1`. |

Three draft steps plus one target/bonus position; built-in MTP head, one linear branch,
classical sampled rejection. Lab sampling sources: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[ZAI](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes). Use lab sampling for model use; comparative tuning screens use temperature0.

## Settings rationale

| Setting | Reason / established limit |
| --- | --- |
| Explicit backends/dtypes/strict | Supported SM70 implementations and precision policy; strict is required during development, not a speed switch. |
| TP/PP/EP and device order | Quad-local NVLink/P2P; cross-quad peer access is unavailable here. PP improves prefill but serializes single-request stages; TP8 MTP decodes faster on Qwen. |
| NCCL PHB/NVLS0, OMP4 | Retained host policy; removing PHB did not help the fresh Qwen screen. NVLS is unavailable on SM70. No universal-optimum claim. |
| Max1/serialized/radix off | Qualified graph/scratch/state ownership and uncached timing. Overlap/concurrency is separate work. |
| Eager prefill/full decode graphs | Qualified native paths; earlier Qwen prefill-graph screens lost. Chunk4352/2048 are measured per-model choices. |
| Context and pool capacity | Qwen targets native262144 plus128page/workspace slots. FP16 PP2/MTP3 allocation,262106-input/32-output generation and request reuse pass. Explicit uncached max1 reservations fail instead of shrinking. |
| Static fraction/state pool | Qwen uses the automatic fraction estimate. GLMTP8.92 is the short reference; GLMPP2MTP.95 qualifies the262K session. The fraction is a profiled allowance, not a physical VRAM allocation or speed setting. With radix off/max1, state slots derive from the request count; a state-memory ratio is redundant. Host PLE saves device memory. |
| GLM attention | `--attention-backend v100_plus` selects the shared native sparse engine, KDA and vision; rows1–4 use bounded split work. Conflicting mode overrides fail. |
| GLM mHC flags0 | Select supported mHC paths; fused DSA top-k retains its default1. |
| CPU-only tests | `CUDA_VISIBLE_DEVICES=""`; numeric/disabled/UUID port allocation is supported. No999 workaround. |

| Optional flag | Recommendation |
| --- | --- |
| `SGLANG_OPT_SM70_HC_PREFILL_SP=1` | Qwen serialized text-only eager prefill, quad-local TP4/TP8/PP2. |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | Qwen PP2 fixed EAGLE1–3 commit copies; GLM fixed chains/full KDA snapshots supported; NVIDIA speed benefit unmeasured, leave off. |
| `SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1` | Qwen TP8 MTP3; leave off on PP2. Unsupported sampling features use their declared original path. |
| `SGLANG_OPT_SM70_NVFP4_MOE_GEMV=1` | GLM opt-in supports1–4 rows; prior publisher screen found no aggregate win, NVIDIA benefit unmeasured. Leave off. |
| `SGLANG_OPT_SM70_NVFP4_GEMV`, `SGLANG_OPT_SM70_MHC_PROJECTION` | Default1 GLM paths; set0 for corresponding unfused controls. |

Qwen TP8 MTP also uses the optional [small-allreduce tuner](v100_plus/nccl/quad_tuner.c):

```bash
mkdir -p .cache
cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so
export NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so" SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1
```

It selects Tree/LL for small eight-rank transfers, leaving large prefill collectives
alone. Forcing Tree globally slows prefill. PP adaptive/DFlash remain guarded;
CP/DCP needs SM70 compression/KV ownership support, and overlap needs buffer leases.
These are implementation gaps, not unused free speed flags.

## Performance and checks

The following history describes frozen `main`; the unified `dev` screen appears
below and uses temperature0. Historical sampled runs are separate controls.

Three uncached coherent near8K requests/profile, one request, eager prefill/full
decode graphs. Current GLM PP2 MTP uses256 sampled outputs,32-token warmups and
fresh A/B/B/A processes (two candidates averaged below), images enabled and262K
context. Other rows retain earlier512-output screens. Qwen MTP rows use fresh
FP16-KV native-context screens;
GLM TP8/ordinary rows remain earlier one-process screens. Seed531 does not guarantee
sampled determinism. PP = input tokens/native prefill seconds; TG = output tokens
excluding the first/native decode seconds. Warmups and profiling are excluded.
These are preliminary throughput measurements. Qwen rows now use FP16 KV;
the earlier E5M2 results are historical controls. Earlier GLM rows:
[ce60d8c9](https://github.com/heislera763/sglang-v100-plus/commit/ce60d8c91c7a0b1c92b1c35059f0bfb7e492d8da)
compatible post-sync paths; NVIDIA PP2 MTP additionally uses the allocator fix
[a85513f5](https://github.com/heislera763/sglang-v100-plus/commit/a85513f5) and exact mHC buffer reuse
[9f057466](https://github.com/heislera763/sglang-v100-plus/commit/9f057466).
GLM PP2 MTP uses [Volta MMA sparse prefill](python/sglang/kernels/ops/attention/v100.py)
and [FP16 routed expert GEMMs](python/sglang/kernels/ops/moe/sm70_fp16.py).
Fresh native A/B/B/A passes on 2026-10-10 gave 1,713 → 2,359 prefill tokens/s
for sparse attention (+37.7%), then 2,357 → 2,548 PP for routed experts (+8.1%).
Small-row attention shortened verification cycles ~3.1%. The subsequent
[live-history indexer](python/sglang/kernels/ops/attention/dsa/sm70_indexer_decode.py)
replaces full-reservation scoring and separate reduction passes: fresh paired
48.1 → 53.2 TG (+10.6%), cycles ~10.2% shorter despite slightly lower acceptance.
Prefill implementation is unchanged (~2,548 PP); baseline prefill varies between
processes, so no prefill gain is attributed to this decode change. Same 262K
context, chunks and FP16 KV. The tables below use the latest confirmed candidate.
[Attention tests](test/registered/kernels/ops/attention/test_triton_sparse_mla_fp16.py) and
[expert tests](test/registered/kernels/ops/moe/test_sm70_fp16.py), [NVFP4 tests](test/registered/kernels/ops/gemm/test_sm70_nvfp4_gemv.py) and
[indexer tests](test/registered/kernels/ops/attention/test_sm70_indexer.py) cover independent
reference math, routing/masks and graph refresh. Precision boundaries are retained;
MMA reduction order is not bit-exact to scalar outputs. Images, 262106-input/32-output
capacity and reuse pass; full-context quality is separate. TP8 below is an earlier
screen preceding these implementations.

The [NVFP4 input-gather patch](v100_plus/patches/marlin-v100-prefill-gather.patch)
caches source rows in registers for each invocation and uses aligned128-bit
loads at64×256×32. On the latest user Marlin baseline, fresh A/B/B/A confirms
**2,902→3,117 PP (+7.4%)**, with category gains7.3–7.6%. Both stage banks are
byte-exact to that baseline; changing routes/counts/masks under graphs,
independent FP64 bounds, canaries and device memcheck pass. MMA/resources and
precision boundaries are unchanged. Verification round wall-time is unchanged;
sampled acceptance varies, so the higher TG below is not a decode speed claim.

The sparse FA prefill integration (2026-10-11) uses a standalone AOT package
from the personal FA fork, preserving selected keys and FP16 KV. Nsight Compute
identified byte-sized CUTE operand copies; fixing their alignment reduced captured
attention time by about 30% and improved near8K model prefill by 4.96–5.28%.
The packaged operator passes reference, CUDA-graph and alignment tests, image
reuse, and 262,106-input/32-output generation followed by request reuse.

A split-K FA decode trial reduced complete attention cost by 23–29%, but model
verification cycles improved only 0–0.54% and sampled generation throughput fell.
Frozen `main` retains its existing decoder. The unified `dev` backend is
qualified separately below; this historical trial is not a current speed claim.

| Model / layout | MTP steps | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| Qwen FP8 TP8/EP8 | 3 | 4,831 | 79.4 |
| Qwen FP8 TP4×PP2/EP4 | 3 | 7,588 | 72.5 |
| NVIDIA GLM TP8/EP1 | 3 | 1,366 | 42.7 |
| NVIDIA GLM TP4×PP2/EP1 | Off | 1,882 | 28.2 |
| NVIDIA GLM TP4×PP2/EP1 | 3 | 3,285 | 59.8 |

MTP3 categories (PP/TG tokens/s; GLM PP2 averages two retained-model processes, others one):

| Model / category | TP8 PP | TP8 TG | PP2 PP | PP2 TG |
| --- | ---: | ---: | ---: | ---: |
| Qwen coding | 4,882 | 82.3 | 7,668 | 82.2 |
| Qwen book continuation | 4,831 | 82.6 | 7,607 | 66.3 |
| Qwen document briefing | 4,779 | 73.3 | 7,489 | 69.0 |
| NVIDIA GLM coding | 1,370 | 48.6 | 3,314 | 66.0 |
| NVIDIA GLM book continuation | 1,367 | 37.6 | 3,275 | 54.8 |
| NVIDIA GLM document briefing | 1,360 | 43.2 | 3,266 | 58.7 |

Inputs: [SPEED-Bench throughput_8k](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
first turns `91d6ca2afe114d3c99312e8758b6f964` (code),
`658dbd96a19e4b138e0aafe43eea1101` (book), `9a936eebf8794621a11f963a56c92120`
(adapted briefing). Qwen8517/8320/8346 input tokens, T1/P.95/K20/xhigh;
GLM7898/8116/8141, T1/P.95/K−1/max. Native metadata via
[return_meta_info](python/sglang/srt/entrypoints/openai/serving_chat.py).
Earlier six-request Qwen ordinary/MTP matrices at
[38f5a0f0](https://github.com/heislera763/sglang-v100-plus/commit/38f5a0f0806883ec88da23ca3c91a46bec7ce12b)
measured TP8 5002/4850PP and50.8/80.2TG; PP2 8078/7619PP and61.9/72.3TG.
Those used E5M2 KV and precede this FP16-cache screen. Current Qwen screens
use HC on, PP2 metadata on, TP8 metadata off, sampling graph/tuner off;
[FP16 QSA numerical/graph tests](test/registered/kernels/ops/attention/qsa/test_sm70_fp16_kv.py) cover the cache port.

Matched independent-process A/B/B/A against [f0245fb0](https://github.com/heislera763/sglang-v100-plus/commit/f0245fb0)
raises GLM PP2 MTP generation 32.2→42.5tokens/s on average: coding+30%,book+30%,
briefing+36%. Prefill is unchanged; speculative cycles are~23% shorter.

The small-row tile changes FP16 reduction rounding; independent FP64 probes found
max absolute error0.00044 versus the reference, old/new difference0.00049. It preserves
selected sparse support and is not bit-exact to the wider tile.

Small-row NVFP4 MMA adds a measured gain over [22e3cffc](https://github.com/heislera763/sglang-v100-plus/commit/22e3cffc):
three control processes and two candidate processes average54.8→59.9generation tokens/s,
with verification cycles6.9% shorter and mean acceptance1.8% higher. Prefill is unchanged.
The kernel uses92registers/18KiB shared versus Marlin192/36KiB;36 captured-bank
routing/graph controls are bit-exact. Independent native-quant FP64 tests include
FP8 scale extremes, changing routes and zero experts; device memcheck reports0errors.
Matching the prefill Marlin row tile to its 64-row routing groups then raises PP
2,545→2,749 (+8.0%) in four alternating processes against
[e23d40aa](https://github.com/heislera763/sglang-v100-plus/commit/e23d40aa).
All four native expert banks remain bit-exact, including graph input/router updates;
memcheck reports zero errors. Decode cycles are unchanged (+0.2%); sampled TG 59.6→58.9
tracks acceptance variation. Long small-head V100 KDA prefills then use two value
lanes per block, raising PP 2,747→2,865 (+4.3%) against
[2d909783](https://github.com/heislera763/sglang-v100-plus/commit/2d909783).
Full-K reductions and FP32 working state remain; captured outputs/final states
and ragged/graph controls are bit-exact, with independent FP64 checks and zero
memcheck errors. Decode cycles remain unchanged (+0.1%). That screen precedes the later prefill changes shown in the table.
Single-request PP decode still serializes stages; receive waiting is not extra
transfer work. Further prefill work centers on expert GEMMs, sparse MMA and KDA.

Other NVIDIA/compact-FP8-MTP PP2 screens rejected mixed64/16/32-row expert
tiles (−5.1% PP), explicit scale/half2 unpacking and narrower-column scheduling.
Eager shared/routed overlap also loses4.4% PP in a three-category near8K/256-token
sampled screen (2,903→2,776). Whole-layer checks are bit-exact, but shared work
mostly overlaps—and slows—the router, barely overlapping the~8.55ms routed branch.
Reversing branch launch order saves less than1% per tested MoE layer; no model
gain is established for that variant. Existing expert kernels and eager scheduling
remain selected.

Explicit shared-tile synchronization then raises current NVIDIA/compact-FP8-MTP
GLM prefill 2,865→2,900 (+1.2%) in matched A/B/B/A processes: coding2,888→2,920,
book2,859→2,892,briefing2,848→2,888 PP. Same near8K/256-token sampled recipe;
the change preserves QK/PV KV reuse and removes redundant per-row gather barriers.
Sparse attention itself takes6.6% less time across both stages; all8 native bank
outputs remain bit-exact. Graph/FP64 controls,16 kernel tests, image/reuse checks
and memory/race/synchronization sanitizers pass. Decode kernels are unchanged;
sampled TG variation is not claimed as a decode gain. Physical-slot sorting and
smaller KV staging were rejected because total operator cost did not improve.

Independent operator bounds plus exact state/payload/RNG checks where specified;
native strict dispatch and EOS/logprob/penalty/cancel/reuse checks supplement them.
Exact full-model token parity across quantizers or TP/PP reductions is not established.
Nsight timelines are separate from timing runs; overlapping GPU durations and
collective waiting are not summed as recoverable wall time.

## Context and multimodal

Long-context and long-video coverage comes from earlier bounded studies; this pass
rechecked Qwen request-level video configuration. These are functional, not broad quality claims.

| Feature | Coverage / configuration |
| --- | --- |
| Qwen1M YaRN | Ordinary PP2, HC/metadata off, `--max-mamba-cache-size 1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; [official RoPE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices): factor4/original262144/theta10000000/partial.25/interleaved MRoPE[11,11,10]. Native977,312-input/512-output functional run; three separated planted codes retrieved. |
| Qwen images/video | `language_model_only:false`, HC/metadata off; SDPA vision. Historical image/H264 and request-level4/8/4-frame checks pass; current unquantized-KV qualification is pending. Request `video_config` overrides startup values without shared mutation. |
| NVIDIA GLM images | PP2/MTP3,262K context allocation,FP16 KV,native D64 FA vision on `dev`. Two swapped images, repeated-image reuse and subsequent8K/512-token text inference pass. Vision weights load only on the first PP stage; image embeddings reach last-stage MTP. Full-context image requests and NVIDIA video remain unqualified. |
| Qwen long video | Startup `--mm-process-config`:2fps/max64frames; optional size budget `{longest_edge:469762048,shortest_edge:4096}`. One-hour fixture sampled full duration; bounded functional coverage, not general video quality. |

Keep original RoPE for short-context benchmarks. Frame/pixel/encoder/KV budgets are
separate limits. Before/after upstream updates run
`uv run --no-project .venv/bin/python v100_plus/check-core-diff.py` and affected checks.

[Apache-2.0](LICENSE). Credit: SGLang, original V100 fork, Marlin/vLLM and model authors;
NCCL declarations retain their [license](v100_plus/nccl/LICENSE.txt).

### Unified GLM attention on dev

Paired-KV prefill screen, 2026-10-13, reusing the matching 2026-10-12 dev reference:
NVIDIA weights, compact FP8 MTP, PP2×TP4/EP1, 24/21 split, 262K FP16 KV/state,
chunk2048, MTP3, near8K coherent prompts, 32-token warmups and 256 outputs.
**Temperature 0** for comparison; earlier lab-sampled measurements remain separate.

| Category | Previous PP | Paired PP | Previous TG | Paired TG |
| --- | ---: | ---: | ---: | ---: |
| Coding |3347|3420|62.6|67.6|
| Writing |3305|3375|54.5|57.1|
| Document analysis |3304|3355|59.1|61.6|

Values are tokens/s. Adjacent queries share an exact multiset of selected keys;
common/exclusive regions avoid redundant loads and masked MMA. A 32-key tile,
separate QK/PV warp partitions and rotated shared-memory layout reduce shared
storage from 82 to 34.4 KiB and permit two resident blocks without spilling.
Complete sparse-operation time falls 20% across eight captured stage/chunk banks,
including support preparation. Whole-model prefill rises 1.5–2.2% in this single
screen, consistent with sparse attention's limited share of total prefill work.
Decode's implementation is preserved. MTP acceptance rose from 75.5/59.4/67.1%
to 82.4/64.8/70.3%; verification-cycle cost stayed within about 1%. The TG changes
therefore do not establish a faster decoder. Main stays frozen.

Selected keys, duplicate multiplicity and FP16 KV are preserved; changed key
ordering and softmax tile boundaries are not bit-identical. Captured outputs
retain `rtol=.001, atol=.0002` bounds against the previous kernel. FP64 support
checks, changing graph inputs, large physical pools, and memory/race/sync
sanitizers pass. The installed FA0.2.1 engine also passes changed/repeated images,
text reuse, and 262106-input/32-output generation followed by request reuse
(2026-10-13). This is capacity coverage, not a long-context quality claim.
Qwen's earlier PP2 smoke was not repeated; its execution paths are unchanged.

The completion pass rejected an extra prefill barrier optimization (0.6% complete
attention gain) and live-only index gathering (large operator gain, no consistent
model TG improvement). A fresh unchanged control also varied in MTP acceptance;
temperature0 does not guarantee identical execution across processes. No additional
prefill or decode throughput gain is claimed. Superseded Triton union/KDA changes
were removed. KDA preparation remains 26–29% faster at operator level; sequential
FP32 recurrence and routed GEMMs remain larger prefill costs. Sparse/KDA kernels
occupy only about4–5% of each stage's decode compute union in the existing trace;
further large decode gains require work beyond the sparse attention kernel.
