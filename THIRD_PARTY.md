# Third-party provenance

At baseline this repository contains **no code from the V100 projects**; they
are candidate sources for selected components.

| Component | Source | License | Status |
|---|---|---|---|
| SGLang | `sgl-project/sglang` | Apache-2.0 | Pinned dependency; not vendored |
| sglang-v100-plus | `heislera763/sglang-v100-plus` | Apache-2.0 | Candidate source; local snapshot at `sglang-restomod-reference` |
| sglang-V100 | `haohervchb/sglang-V100` | Apache-2.0 (verify) | Candidate source |
| sglang-sxm2 | `dg1kjd/sglang-sxm2` | Apache-2.0 (verify) | Source of `attention_v100` and the long-context decode kernel |
| flash-attention-v100 | `heislera763/flash-attention-v100` | see upstream | Candidate kernel dependency |
| marlin_v100 | `zhinianqin/marlin_v100` | see upstream | SM70 Marlin kernels; built separately for NVFP4 and block-FP8 expert GEMM |
| 1Cat-vLLM | `1CatAI/1Cat-vLLM` | see upstream | FP8 expert GEMM contract; not vendored |
| TurboMind | `InternLM/lmdeploy` | Apache-2.0 | Prebuilt SM70 library bound by `fp8_moe`; not vendored |
| CUTLASS | `NVIDIA/cutlass` | BSD-3-Clause | Build dependency of the Marlin extension (v4.2.1) |

Rules:

- Every selected component records its source repository, commit, and license
  in the PR and in the file header or manifest entry.
- Retain the upstream `LICENSE`/`NOTICE` when redistributing.
- This file is informational, not legal advice.
