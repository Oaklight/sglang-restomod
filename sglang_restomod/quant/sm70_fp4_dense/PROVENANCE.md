# Provenance: quant/sm70_fp4_dense

## Source

| Field | Value |
|---|---|
| Repository | `zhinianqin/marlin_v100` |
| Commit | `3f16d442cdb4c24dd225bbec196c982a54d9a31c` |
| License | Apache-2.0 by lineage (vLLM and Marlin); the repository ships no `LICENSE` file |
| Built artifacts | `_sm70_marlin_v100_dense.abi3.so` (sm_70) for this path; `_moe_C` stays with `fp4_moe` |

No new source is vendored and no CUDA is added. This component binds the two
operators the build already exports -- `gptq_marlin_repack` and `marlin_gemm`
-- and reuses the argument mapping `quant/sm70_marlin_v100` established for the
19-slot `marlin_gemm` signature.

**Which build.** Point `SM70_MARLIN_DIR` at a directory that symlinks the
prebuilt dense library as `_C.so`:

```sh
mkdir -p ~/marlin-ops && cd ~/marlin-ops
ln -sf .../prebuilt/_sm70_marlin_v100_dense.abi3.so _C.so
```

A smaller schema-only `_C.abi3.so` also loads and exports the name, but has no
CUDA implementation, so the first GEMM fails with `Could not run
'_C::marlin_gemm' with arguments from the 'CUDA' backend`. The check in
`check.py` names both builds.

## What this component uses

| Symbol | Bound as | Why |
|---|---|---|
| `_C.gptq_marlin_repack` | `marlin_utils_fp4.gptq_marlin_repack` | SM70 tile layout: the weight must be packed by the build that reads it |
| `_C.marlin_gemm` | `marlin_utils_fp4.gptq_marlin_gemm` | the dense NVFP4-A16 GEMM; `b_q_type.id` selects `float4_e2m1f` |
| (none) | `marlin_utils_fp4.marlin_permute_scales` | identity: the SM70 iterator reads unpermuted scales |
| (none) | `marlin_utils_fp4.nvfp4_marlin_process_global_scale` | collapses a fused layer's per-shard `weight_scale_2` with `max()`, then converts to fp16 |

## Kernel preconditions

Enforced inside the build (`csrc/quantization/marlin/marlin.cu:454-476` of the
commit above); recorded here so a shape change is caught before a wrong answer:

- weights E2M1, block scales `float8_e4m3fn` (NVFP4) or `float8_e8m0fnu` (MXFP4);
- activations and output at 16 bits;
- `size_k % 32 == 0`, `group_size == 16`, no zero-point;
- a scalar `global_scale` -- hence the fourth patch.

## Upstream facts this relies on

At sglang `v0.5.22`:

- `modelopt_quant.py:2052` -- `ModelOptFp4LinearMethod.apply` returns into
  `apply_fp4_marlin_linear` when `--fp4-gemm-backend marlin`, before any
  activation quantization. A checkpoint layer tagged `NVFP4` therefore runs as
  W4A16 here; no `quant_algo` rewrite and no synthetic `W4A16_NVFP4` tag.
- `marlin_utils_fp4.py:22` -- both kernel names are imported into that module's
  namespace, which is why patching `marlin_utils` (what `sm70_marlin_v100`
  does) does not reach the FP4 path.

Both are re-checked at install time: a missing name fails the install rather
than silently leaving the pin's own kernel in place.
