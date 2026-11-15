# Provenance: quant/sm70_mtp_fp4_moe

## Source

No third-party code is added by this component. It emits the NVFP4 checkpoint
layout that the engine and `quant/sm70_fp4_moe` already handle, and it calls the
same `ModelOptNvFp4FusedMoEMethod`. The Marlin extension and its licence are
recorded in `quant/sm70_fp4_moe/PROVENANCE.md`.

| Field | Value |
|---|---|
| Kernel | `_moe_C.moe_wna16_marlin_gemm` from `zhinianqin/marlin_v100` @ `3f16d442cdb4c24dd225bbec196c982a54d9a31c`, Apache-2.0 by lineage |
| Scale encoding | `sglang_restomod.quant.sm70_fp4_moe.scales` (unchanged) |
| Layout reference | `sglang.srt.layers.quantization.dequantization.dequantize_nvfp4` at the pin |

## Why the draft never reached the kernel

Two engine facts, both at the pin, and together they are the whole explanation:

1. `modelopt_quant.py` — the quant-method lookup returns `None` for a
   `FusedMoE` whose prefix the checkpoint excludes. This checkpoint's
   `quantization_config.ignore` lists `mtp.*` and `model.mtp.*`, so the draft's
   expert layer is `UnquantizedFusedMoEMethod`.
2. `unquant.py` — that method constructs `MoeRunner(MoeRunnerBackend.TRITON,
   ...)` unconditionally. `--moe-runner-backend marlin` is not consulted
   anywhere in that file.

So the draft's experts were on Triton **by construction**. No launch flag could
reach the Marlin GEMM while the layer stayed unquantized.

## How aggressive this is, against the precedents

Quantizing a draft's expert weights to 4 bits is more aggressive than any
precedent in the comparison set. Recorded here so a later accept regression has
somewhere to point:

- **This checkpoint** excludes `mtp.*` from NVFP4. That is ModelOpt's default
  export and is visible in `model-sm70/config.json` (verified here).
- **DeepSeek's FP8 checkpoints** keep MTP in BF16 rather than FP8 (reported by
  the project coordinator from the comparison set; not re-verified here).
- **1Cat's draft** — the only quantified draft in the comparison set — is FP8,
  not 4-bit (same provenance: reported, not re-verified here).

The acceptance gate is therefore the draft's accept rate, not the step time: a
faster draft that accepts less is a trade, and it is reported as one.

## Separability

The ladder below is cheap to walk because the two halves are separate files:

| Step | File | Replaceable independently |
|---|---|---|
| Quantize the bank to a checkpoint layout | `quantize.py` | Yes — emit FP8 (block 128) instead and keep the swap |
| Give the layer those tensors and the NVFP4 method | `__init__.py` | Yes — it never inspects the format |
| Kernel and repack | `quant/sm70_fp4_moe` | Unchanged, shared with the target |

Fallback order if 4-bit costs accept: FP8 (matching 1Cat), then FP16 with better
tiles (that last one is not this component's problem).

## What this component does not touch

The target model. Selection is by observed state inside a model whose class is
the MTP class: a `FusedMoE` still on `UnquantizedFusedMoEMethod`. The target's
experts are NVFP4 and never match, and a launch that does not select this
component is byte-for-byte the previous behaviour.

It also refuses to convert unless `fp4_moe` is installed, because the pinned
Marlin kernel is an empty-body stub below SM80: converting without the
replacement would zero the draft's experts instead of speeding them up.
