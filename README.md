# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) adapted for **eight V100 SXM2 32 GB GPUs
in two NVLink quads**. We rebuilt useful SM70 support from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
effectively rebasing it into an independent fork. Qwen3.8-Flash-Next FP8 and
GLM-5.3-Flash NVFP4 work is consolidated on `main`.

`main` contains validated changes; `dev` is the integration branch. Both include
the qualified unified GLM attention work. The [FA fork](https://github.com/heislera763/flash-attention-v100)
owns native CUDA sparse, vision and KDA attention; SGLang owns thin adapters,
metadata and cache/state lifetimes. Advance `main` after correctness and model
throughput checks. Qwen retains its existing backend during this work.
Clone remotes: `origin` is this fork; `upstream` is `sgl-project/sglang`.

Integrated upstream: [ab580b2e](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), integrated 2026-10-19.
The merge includes DSA index-cache relocation, speculative recurrent-checkpoint
alignment, pipeline residual-handoff and MoE scaling correctness fixes. Shared-expert
layouts are now immutable per-model configuration, scoped for target/draft use.
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
| Local GLM MTP FP8 | NVIDIA source above | Routed MTP experts converted to128×128 E4M3 block FP8/W8A16; other tensors unchanged. Used by current GLM measurements; local `PROVENANCE.json` records conversion. | 7.10 GiB |

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
across TP ranks. The current GLM PP2/MTP3 session limit is **524,288** with a
524,800-slot pool (~3.10GiB/GPU including target/draft KV and index storage).
A 524,250-input/32-output capacity probe, subsequent request reuse and an image-bearing
near8K request pass with FP16 KV/state and compact FP8 MTP. This establishes functional
capacity, not long-context quality. Native1M remains unqualified. TP8 retains its
262,144-context/262,656-slot profile (~3.10GiB/GPU); larger TP8 windows are unqualified.

## SM70 execution

Volta supports FP16 Tensor Cores. FP4/FP8/BF16 checkpoint storage is decoded or cast
in software; it does not imply native arithmetic in those formats.

| Operation | Implementation / contract |
| --- | --- |
| NVFP4 GEMMs | [marlin-v100](https://github.com/heislera763/marlin-v100) at `95b1e62e` plus local patches: W4A16, FP16 WMMA with FP32 accumulation; block/global scales remain distinct. TP4/TP8 routed prefill uses64×256×32 cached-row Marlin tiles. Small routed groups at both TP sizes use [Volta MMA](python/sglang/kernels/ops/gemm/sm70_nvfp4_routed.py) with M16 tiles and K-major decoded weights; FP16 dequantization/output and FP32 accumulation/scaling remain. Unused alternate-backend MoE scale buffers alias active scales, saving17.72GiB across GLM's8GPUs. |
| Block-FP8 GEMMs | Patched Marlin and small-row vector kernels: W8A16, complete 128×128 scale blocks. Qwen EP4/EP8 preserves whole experts. |
| FP16 routed experts | [Volta MMA GEMM](python/sglang/kernels/ops/moe/sm70_fp16.py), used by NVIDIA GLM's unquantized MTP experts. FP16 inputs/weights/output, FP32 accumulation and router multiply; upstream routing, activation and expert sum retained. Automatically selected by the SM70 plugin; unsupported epilogues fail under strict dispatch. |
| Qwen QSA/GDN/HC | Direct FP16-cache sparse prefill/decode/verification, masked Tensor Core prefill, SM70 GDN and gated-residual kernels; HC eager prefill partitions within each quad, including TP8. |
| GLM attention | [Canonical V100 CUDA backend](python/sglang/kernels/ops/attention/v100.py) from the [FA fork](https://github.com/heislera763/flash-attention-v100): selected-key FP16 sparse prefill/decode/MTP, FP32 reductions and FP16 probabilities/partials. Empty tiles skip identity work. The model-native FP8 indexer retains [live-history MMA scoring](python/sglang/kernels/ops/attention/dsa/sm70_indexer_decode.py) and bounded prefill scoring. |
| GLM KDA / mHC | Native CUDA full-K FP32 recurrence, FP16 state and per-step verification carry. KDA prefill prepares normalized Q/K and gates once; a Tensor Core chunk formulation is future work. Bulk mHC shares statistics/mixing within each quad, gathering FP16 input and FP32 coefficients. Its learned FP32 projection stays replicated to preserve reduction order. |
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

The GLM attention profile uses one selector, `--attention-backend v100_plus`,
for sparse, KDA and vision. It requires FP16 compute/state, compute-dtype `auto`
KV, `no_buffer` or `extra_buffer` recurrent caching and one running request. Conflicting backend
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
  --default-chat-template-kwargs '{"clear_thinking":false}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

| Variation | Change from reference |
| --- | --- |
| Qwen TP8 | TP8/EP8; omit PP and unset `SGLANG_PP_LAYER_PARTITION`/metadata flag. Keep HC/chunk4352. |
| GLM TP8 MTP | TP8/EP1,262144context/262656pool/.95fraction/chunk2048 and the compact draft flags below. Export the small-allreduce tuner below and `SGLANG_OPT_SM70_HC_PREFILL_SP=1`. |
| GLM PP2 MTP | TP4/PP2/EP1; `SGLANG_PP_LAYER_PARTITION=24,21`, `SGLANG_OPT_SM70_HC_PREFILL_SP=1`; context **524288**, total tokens **524800**, fraction **.95**, chunk2048. Add MTP flags below and `--speculative-draft-model-path "$HOME/models/sglang/local-GLM-5.3-Flash-MTP-FP8" --speculative-draft-model-quantization fp8`. Images are enabled; vision weights reside on the first PP stage. The table below leaves metadata graphs and the small-allreduce tuner off for PP2. |
| MTP, either model | Add `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 --speculative-use-rejection-sampling`. PP2 also exports `SGLANG_ENABLE_PP_SPEC=1`. |

Three draft steps plus one target/bonus position; built-in MTP head, one linear branch,
classical sampled rejection. Lab sampling sources: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[ZAI](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes). Use lab sampling for model use; comparative tuning screens use temperature0.

The commands above disable prefix reuse for uncached comparisons. The qualified
GLM PP2 conversation profile uses these changes:

| Setting | Value / reason |
| --- | --- |
| Prefix caching | Remove `--disable-radix-cache`; use `--mamba-radix-cache-strategy extra_buffer`. KV and recurrent checkpoints remain FP16. |
| State capacity | `--max-mamba-cache-size 4`: bound one active session, tracking and retained-state headroom; keep the524800-token pool. |
| Checkpoint placement | `--mamba-track-interval 2048`: apply to prefill and decode, matching the prefill chunk. Avoid introducing an interior FP16 rounding boundary when resuming the FP32 prefill recurrence. |
| Agent history | `clear_thinking=false`; return complete original reasoning blocks for prefix continuity. |
| Pi client | `maxTokens:131072`, `samplingParams:{"temperature":1.0,"top_p":0.95}`. Output and input share the524288-token window. |

A repeated7898-token/128-output screen reused6144 input tokens: effective
prefill3142→8584tokens/s and generation63.6→62.2tokens/s. Effective prefill
counts the reused prefix; it is not an attention-kernel throughput gain.
Text/tool/image reuse checks pass, including bit-exact fixed first-token
top32 logprobs. Full greedy MTP sequences can vary; independent uncached
controls also show numerical variation, so sequence identity is not guaranteed.

## Settings rationale

| Setting | Reason / established limit |
| --- | --- |
| Explicit backends/dtypes/strict | Supported SM70 implementations and precision policy; strict is required during development, not a speed switch. |
| TP/PP/EP and device order | Quad-local NVLink/P2P; cross-quad peer access is unavailable here. PP improves prefill but serializes single-request stages; TP8 MTP decodes faster on Qwen. |
| NCCL PHB/NVLS0, OMP4 | Retained host policy; removing PHB did not help the fresh Qwen screen. NVLS is unavailable on SM70. No universal-optimum claim. |
| Max1/serialized/cache policy | Qualified graph/scratch/state ownership. References use radix off for uncached timing; GLM PP2 conversations enable the bounded FP16 prefix cache above. Overlap/concurrency is separate work. |
| Eager prefill/full decode graphs | Qualified native paths; earlier Qwen prefill-graph screens lost. Chunk4352/2048 are measured per-model choices. |
| Context and pool capacity | Qwen targets native262144 plus128page/workspace slots. FP16 PP2/MTP3 allocation,262106-input/32-output generation and request reuse pass. GLM PP2/MTP3 reserves524800 slots for524288 context;524250-input/32-output, images and reuse pass. Explicit max1 reservations fail instead of shrinking, including with prefix reuse enabled. |
| Static fraction/state pool | Qwen uses the automatic fraction estimate. GLM uses.95 for262K TP8 and524K PP2 MTP profiles; .92 is the earlier short reference. The fraction is a profiled allowance, not a physical VRAM allocation or speed setting. With radix off/max1, state slots derive from the request count; cached GLM conversations pin4 slots. A state-memory ratio is redundant in both cases. Host PLE saves device memory. |
| GLM attention | `--attention-backend v100_plus` selects the shared native sparse engine, KDA and vision; rows1–4 use bounded split work. Conflicting mode overrides fail. |
| GLM mHC flags0 | Select supported mHC paths; fused DSA top-k retains its default1. |
| CPU-only tests | `CUDA_VISIBLE_DEVICES=""`; numeric/disabled/UUID port allocation is supported. No999 workaround. |

| Optional flag | Recommendation |
| --- | --- |
| `SGLANG_OPT_SM70_HC_PREFILL_SP=1` | Serialized eager prefill within TP4/TP8/PP2 quads. Qwen partitions HC residuals; GLM partitions bulk mHC statistics/mixing at≥1024rows, restoring full input/coefficients before attention and experts. Small rows and decode keep their existing native paths. |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | Qwen PP2 fixed EAGLE1–3 commit copies; GLM fixed chains/full KDA snapshots supported; NVIDIA speed benefit unmeasured, leave off. |
| `SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1` | Qwen TP8 MTP3; leave off on PP2. Unsupported sampling features use their declared original path. |
| `SGLANG_OPT_SM70_NVFP4_MOE_GEMV=1` | GLM opt-in supports1–4 rows; prior publisher screen found no aggregate win, NVIDIA benefit unmeasured. Leave off. |
| `SGLANG_OPT_SM70_NVFP4_GEMV`, `SGLANG_OPT_SM70_MHC_PROJECTION` | Default1 GLM paths; set0 for corresponding unfused controls. |

TP8 uses the optional [small-allreduce tuner](v100_plus/nccl/quad_tuner.c):

```bash
mkdir -p .cache
cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so
export NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so"
```

It selects Tree/LL for small eight-rank transfers, leaving large prefill collectives
alone. On GLM it shortens verification cycles about 2%; large-transfer protocol
controls found no prefill gain. Qwen sampling graphs use the separate optional
flag listed above. PP adaptive/DFlash remain guarded;
Upstream DSA/MTP DCP and logical-page KV sharding are available as integration
work: the V100 backend still needs compatible partial-attention/LSE and KV
ownership support. At524K the sharding default reserves about8GiB of assembly
scratch per GPU (two slots, eight contexts each); single-user sizing needs
attention before enabling it. Overlap needs buffer leases.
These are implementation gaps, not unused free speed flags.

## Performance and checks

GLM upstream-integration screen (2026-10-19): NVIDIA NVFP4 target, compact FP8
MTP3, images, FP16 KV/state, max1, chunk2048, eager prefill/full decode graphs,
strict dispatch and FP16 prefix reuse (4 recurrent slots, track2048,
`clear_thinking=false`). PP2×TP4/EP1 uses24/21 layers and524288 context/524800
slots; TP8/EP1 uses262144/262656 and the small-allreduce tuner. **Temperature0**,
32-token warmups and256 outputs per coherent near8K case; caches flushed before
each request. Recorded controls match each layout's capacity and settings;
PP2's524K control was collected immediately before integration. Results cover the
final integrated source; compilation/warmup and profiling are outside timed requests.

| Category, tokens/s | PP2 prefill | PP2 generation | TP8 prefill | TP8 generation |
| --- | ---: | ---: | ---: | ---: |
| Coding |3,127|59.3|2,208|87.5|
| Writing |3,096|51.8|2,206|66.1|
| Document analysis |3,092|60.4|2,205|78.2|

| Tokens/s, category means | Before upstream merge | After |
| --- | ---: | ---: |
| PP2 prefill |3,080|3,105|
| PP2 generation |57.4|57.2|
| TP8 prefill |2,178|2,206|
| TP8 generation |79.7|77.3|

This short screen establishes no large throughput gain. TP8 mean MTP acceptance
changes from73.1% to69.8%, with variation across prompts; verification-cycle
wall time stays effectively unchanged. The acceptance difference is not isolated
to a particular upstream commit. Historical optimization screens below use earlier
cache/template policies.

Earlier mHC screens improved prefill **2.3% for PP2 and2.8% for TP8**. Whole
PRE/gather/POST on two stage banks improves about 19% at 2048 rows and remains bit-exact to
the preceding FP32/FP16 implementation. Quarter-row projection was rejected
because it changed cuBLAS reduction order; the projection and full residual
remain replicated. Images, near8K image-bearing input, request reuse and the
262106-input/32-output PP2 capacity check pass. No decode gain is attributed:
cycle means 50.60→50.31 ms PP2 and 39.91→39.90 ms TP8; throughput varies with acceptance.

Values are tokens/s. TP8 starts at 2,034 prefill / 67.8 generation (category means).
Extending the existing M16 routed NVFP4 path to TP8 shortens verification
cycles 10.5%; matching prefill route groups to the 64-row cached gather raises
prefill 9.4%. The size-limited tuner adds about 2% shorter cycles. Together the
screen gives **2,034→2,226 prefill** and **67.8→78.8 generation** (+16.2%).
Temperature0 does not make floating-point execution or MTP acceptance identical
across processes. PP2 remains faster at prefill; TP8 uses both quads concurrently
during decode.

Quad-local attention was tested inside TP8 with attention DP2/TP4 and distributed
experts. EP2×expert-TP4 produced2,002 prefill /56.9 generation; EP8×expert-TP1
produced1,935/55.1. Both lose to the matching2,226/78.8 TP8 control; execution
cycles also lengthen. Attention locality alone does not remove distributed
expert transport, padded scheduling or expert compute costs. These layouts stay
unqualified; no extra dispatch branches or tuning flags were retained.

Selected-key paired attention is unchanged in this pass. The canonical FA
engine shares exact selected-key multisets between adjacent queries, uses
FP32 online softmax/accumulation with FP16 KV/probabilities/output, and runs
native KDA and D64 vision. Paired sparse prefill reduces complete operator time
20% and improved the earlier matching PP2 model screen 1.5–2.2%. Sparse/KDA
currently have a small decode budget; routed/dense GEMMs and communication
account for more time. NVLink carries quad-local communication; cross-quad
NCCL uses host shared memory. Collective durations include rank arrival waits.

Decode counters identify underfilled native Volta FP16 GEMMs, not a newer-GPU
fallback: KDA's fused projection reaches 4.1% occupancy/47% DRAM throughput;
shared gate/up reaches 6.2%/26% and launches 64 blocks across 80 SMs. Both use FP32
Tensor Core accumulation. FP32 split-K trials are rejected: shared gate/up's
finishing pass erases its GEMM saving; KDA's local saving does not improve model
throughput. TP8 before/KDA-only means are 2,289/78.1→2,286/77.9 prefill/generation;
the combined trial gives 2,289/75.6. Generation span per verification cycle is
39.90→40.08 ms for KDA alone. Wider TP4 banks also lose. TP8 FP64/cuBLAS, graph
refresh, canary, image/reuse and memcheck checks pass at unchanged numerical
bounds. Existing cuBLAS and the retained results above remain unchanged.

TP8 qualification: both stage-derived real weight banks, gate/up and down,
changing routes/counts/router/inputs under graph replay, masked/empty experts,
alignment and output canaries. Prefill is byte-exact to current Marlin; small-row
gate/up is bit-exact; eager down sub-bank differences are ≤6.1e-5. Independent
native-quant FP64 and graph checks retain `rtol=.001, atol=.0002`; memory sanitizer reports zero
errors. Swapped/repeated images and subsequent text reuse pass. Canonical adapters
now preserve token-major sparse output and trim snapshot indices to current
sequence capacity. Graph capture uses the smallest aligned request bucket and
the resolved draft-extension width. Live graph, zero-tail and snapshot-canary
regressions pass with zero memory-sanitizer errors. Unsupported
attention DP/CP configurations fail at profile resolution. Kernel tests:
[NVFP4 GEMM](test/registered/kernels/ops/gemm/test_sm70_nvfp4_gemv.py),
[expert/gather](test/registered/kernels/ops/moe/test_sm70_nvfp4.py),
[adapter contracts](test/registered/kernels/ops/attention/test_v100_backend.py),
[canonical attention](https://github.com/heislera763/flash-attention-v100/blob/2345e3da21c6501a26f691b9325ded12ab34e52b/csrc/sparse_mla/tests/test_attention.py).
Narrower tiles and six unpacking/scheduling prototypes lost their operator
screens and were rejected without redundant model runs. Large-collective
protocol overrides also failed the screen.

Inputs: pinned [SPEED-Bench throughput_8k](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
first turns `91d6ca2afe114d3c99312e8758b6f964` (code),
`658dbd96a19e4b138e0aafe43eea1101` (book),`9a936eebf8794621a11f963a56c92120`
(adapted briefing); GLM 7,898/8,116/8,141 input tokens. PP=input/native prefill
seconds; TG=(output−1)/native decode seconds, from
[return_meta_info](python/sglang/srt/entrypoints/openai/serving_chat.py).
These are preliminary single-request screens, not concurrent-serving numbers.

Qwen paths are unchanged. Its earlier lab-sampled FP16-KV/MTP3 screens use
T1/P.95/K20/xhigh, 512 outputs and 8,517/8,320/8,346-token inputs:

| Qwen FP8 profile | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: |
| TP8/EP8 |4,831|79.4|
| TP4×PP2/EP4 |7,588|72.5|

These sampled results are separate from the GLM temperature0 comparison.
Historical GLM screens and intermediate optimization attribution remain in the
[frozen main README](https://github.com/heislera763/sglang-v100-plus/blob/f979892c05607f944249127035b8dff6c0f00322/README.md#performance-and-checks).

## Context and multimodal

Long-context and long-video coverage comes from earlier bounded studies; this pass
rechecked Qwen request-level video configuration. These are functional, not broad quality claims.

| Feature | Coverage / configuration |
| --- | --- |
| Qwen1M YaRN | Ordinary PP2, HC/metadata off, `--max-mamba-cache-size 1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; [official RoPE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices): factor4/original262144/theta10000000/partial.25/interleaved MRoPE[11,11,10]. Native977,312-input/512-output functional run; three separated planted codes retrieved. |
| Qwen images/video | `language_model_only:false`, HC/metadata off; SDPA vision. Historical image/H264 and request-level4/8/4-frame checks pass; current unquantized-KV qualification is pending. Request `video_config` overrides startup values without shared mutation. |
| NVIDIA GLM images | PP2/MTP3,524K context allocation,FP16 KV,native D64 FA vision on `dev`. Historical swapped-image/repeated-image checks pass; current524250-input/32-output capacity run is followed successfully by text reuse and an image-bearing near8K request. Vision weights load only on the first PP stage; image embeddings reach last-stage MTP. Full-context image requests and NVIDIA video remain unqualified. |
| Qwen long video | Startup `--mm-process-config`:2fps/max64frames; optional size budget `{longest_edge:469762048,shortest_edge:4096}`. One-hour fixture sampled full duration; bounded functional coverage, not general video quality. |

Keep original RoPE for short-context benchmarks. Frame/pixel/encoder/KV budgets are
separate limits. Before/after upstream updates run
`uv run --no-project .venv/bin/python v100_plus/check-core-diff.py` and affected checks.

[Apache-2.0](LICENSE). Credit: SGLang, original V100 fork, Marlin/vLLM and model authors;
NCCL declarations retain their [license](v100_plus/nccl/LICENSE.txt).
