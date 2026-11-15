# 2026-11-06 — Qwen4 27B on V100: no checkpoint exists

Status: **research note, no GPU run.** As of 2026-11-06 a Qwen4 27B cannot be
assessed on any hardware: no model card, config, license, checkpoint or HF repo
exists. The newest released 27B is Qwen3.8-27B, and it is already served here.

Recorded from issue #231. Full arithmetic and the lab note:
<https://github.com/Oaklight/lambda-v100-lab/blob/notes/qwen4-27b/notes/qwen4-27b-v100-assessment-2026-11-06.md>

## What the primary sources said

| Check, 2026-11-06 | Result |
|---|---|
| Alibaba Cloud press release, Hangzhou, 2026-09-22 | "Qwen 4, is currently in training". No tiers, no 27B, no weights, no date. Qwen 4.5 / Qwen 5 projected at 5–10T params. |
| HF API, `author=Qwen`, 100 most recent repos | no `Qwen4`/`qwen-4` repo. Newest 27B is `Qwen/Qwen3.8-27B`, created 2026-08-05. |
| HF API, `search=qwen4` | third-party repos only, e.g. a GGUF of Qwen3.8-Flash-Next tagged qwen4. |

Secondary coverage reporting a four-tier preview (Max, Plus, Flash, an
open-weights 27B) is not in the press release and was not treated as fact.

## Qwen3.8-27B, the serveable 27B

From the model card and `config.json`, fetched 2026-11-06:

| Field | Value |
|---|---|
| Params / stage | 27B, dense, post-trained |
| Modality | native vision-language (`Qwen3_5ForConditionalGeneration`, vision depth 27) |
| Hidden / FFN | 5120 / 17,408 |
| Layers | 64 = 16 x (3 x GDN → FFN, 1 x gated attention → FFN) |
| GDN | 48 layers, 16 QK / 48 V heads, head dim 128, conv kernel 4, state fp32 |
| Gated attention | 16 layers, 24 Q / 4 KV heads, head dim 256, partial rotary 0.25 |
| Context | 262,144 native; 1M with YaRN |
| MTP | 1 MTP layer in config |
| Vocab | 248,320 |
| License | Apache-2.0 |

Checkpoint sizes: bf16 55.6 GB (18 shards), FP8 29 GB, ModelOpt NVFP4 (HF
only), unsloth NVFP4 (HF only), NVFP4 RadixArk/QUASAR 21/20 GB, W4A16 AutoRound
19 GB, GGUF UD-Q4_K_XL / UD-IQ3_XXS 17/11 GB, DFlash2 draft 3.6 GB / Q8_0 2 GB.
Qwen4-27B in any format: nowhere.

## Fit at 32 GB per card, inference only

Weights per card, TP2 / TP4 / TP8: bf16 27.8 / 13.9 / 6.9 GB; FP8 14.5 / 7.2 /
3.6; NVFP4 ideal 7.6 / 3.8 / 1.9; shipped NVFP4 10.5 / 5.2 / 2.6; W4A16 9.5 /
4.8 / 2.4.

KV is 64 KiB per token in fp16 (16 full layers x 4 KV heads x 256) and 32 KiB in
e5m2: 2 / 8 / 16 GiB total at 32k / 131k / 262k, halved for e5m2. GDN state is
per sequence, 151 MB fp32 (75 MB bf16), sharded with TP.

- TP2 + FP8 + 262k fp16 ≈ 22.7 GB per card: fits.
- TP2 + bf16 is ~28.9 GB at 32k with fp16 KV (tight) and 35.5 GB at 262k (does
  not fit); 29.9 GB at 131k with e5m2.
- TP4 is comfortable in every format; bf16 + 262k fp16 ≈ 17.9 GB.
- TP8 adds little: 4 KV heads cap KV sharding at TP4.
- 1M context needs 15.3 GiB per card of e5m2 KV at TP2, only on FP8 weights.

## Framework support at the pin v0.5.22 (`baa9befa`)

`qwen3_5` (model, text, MTP, config) is in the pin, so Qwen3.8-27B loads and is
measured here. `qwen4_exp` is in the pin as the Qwen4-generation architecture,
with one end-to-end test on B200 for the NVFP4 Flash-Next checkpoint; that is
the Qwen4-generation model that exists.

Components for a dense 27B: `attention_v100`, `fp8_dense`, `fp8_dense_quant`,
`fp8_lm_head`, `gdn_tilelang_sm70`, `spec_sm70`. `fp4_moe` does not apply;
dense NVFP4 on SM70 is #178 and not production-ready.

## Measured here, Qwen3.8-27B on V100

44.4–57.1 ms/step at 4.3k–31.5k (TP2 FP8, MTP); 684 tok/s prefill at 31k with
the Triton GDN and 1,650 tok/s with `gdn_tilelang_sm70`; TTFT 75 s at 131k.
e4m3 KV is unavailable on Volta and e5m2 costs about 1.4x decode. 1Cat-LLM on
the same box: 105–124 tok/s decode, 2,373 tok/s prefill at 8k.

## Blockers for the released 27B, all measured

1. Decode against 1Cat: about 2x at the same weights and hardware, from the
   dense-GEMM share and verify attention. This is the #201/#224 line.
2. KV dtype at long context: e4m3 is unavailable on Volta, e5m2 costs ~1.4x.
3. Prefill at long context: 1,650 tok/s at 31k against 1Cat's 2,333.

## Effort if a Qwen4-27B lands

| case | effort |
|---|---|
| `qwen3_5`-shaped dense 27B | days: load, quantize, rerun the 27B recipes |
| `qwen4_exp`-shaped dense | 2–4 weeks of glue |
| new attention or quant op | unknown until a config exists |

Sourced: the press release, the HF absence checks, every Qwen3.8-27B field, all
sizes, the pin's file list, and the lab measurements. Inferred: the per-card
tables, KV and state arithmetic, TP8 KV replication, effort estimates, and
anything about a future Qwen4-27B.
