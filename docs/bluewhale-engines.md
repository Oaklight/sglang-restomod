# Which checkpoint, which engine

A map of the DeepSeek-V4 family we hold, what each artifact actually is, and
which loading path in our pinned SGLang can accept it. Written after a bring-up
lost a night to the boundary, which is the reason it exists: the two models
differ by `.1` in their names and by a different loader underneath.

## Two models, not one version string

| model | size | experts | GGUF architecture label | published formats |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | 284B total, 13B active | — | `deepseek4` | **MXFP4 only** upstream |
| DeepSeek-V4.1-Flash-1017 (`bluewhale`) | 40 layers, hidden 5120 | 384 routed, top-6, inter 2304 | `deepseek41` | safetensors fp8 + MXFP4 experts; llama.cpp I-quants |

Everything below follows from one sentence: **the artifact and the engine path
have to match in both container and quantisation**, and these two models do not
match each other in either.

## Artifacts we hold

| path | model | container | routed experts | dense | usable by |
| --- | --- | --- | --- | --- | --- |
| `models/bluewhale/base-fp8/` | V4.1 | safetensors, 510 GB | MXFP4: E2M1 two per byte, E8M0 scales on a block-32 grid (`weight_block_size [32,32]`, `expert_dtype fp4`) | fp8 e4m3, ue8m0, `[32,32]` | SGLang safetensors path (with work) |
| `models/bluewhale/_upstream/bluewhale-gsq-iq1m/` | V4.1 | GGUF, 120.7 GB + 110.6 GB Engram | IQ1_M (1.75 bpw) | GGML in-file | Strata |
| `models/bluewhale/_upstream/bluewhale-gsq-iq2xs/` | V4.1 | GGUF, 265.8 GB + Engram | IQ2_XS (2.25 bpw) | GGML in-file | Strata |
| `models/bluewhale/iq2xs-pack/` | V4.1 | Strata pack (`ds4.txt`, `tensors.txt`, `engram.bin`, tokenizer) | by index | by index | Strata |

The publisher offers three sizes — IQ1_M, IQ2_XXS, IQ2_XS — and **no MXFP4**
GGUF (verified against the repository's file list on 2026-11-16).

## Engine paths and their gates

### Strata — the artifact-engine pair that exists

Takes the ISTA GGUF plus its pack and serves on V100 today. This is the path
`iq2xs-pack/` was built for.

### SGLang expert pack — needs MXFP4 experts in a GGUF

| gate | evidence | consequence |
| --- | --- | --- |
| expert dtype must be MXFP4, block 32, `identity-v1` | `srt/layers/moe/expert_pack.py:301-303` raises `unsupported expert dtype` | I-quant GGUFs are rejected at store init — after the pack has been written |
| the source must be a GGUF | `srt/model_loader/expert_pack_loader.py:228-229` raises `DeepSeek source GGUF is missing`; non-experts come from `deepseek4_nonexpert_weights_iterator(source_path)` | a safetensors tree (base-fp8) cannot feed it, even though its experts *are* MXFP4 |
| the header must say `deepseek4` and carry `deepseek4.*` | `srt/model_loader/expert_pack_runtime.py:257`, dispatcher at `:598` | a V4.1 (`deepseek41`) GGUF needs an alias |
| no `general.sampling.*` in the V4.1 GGUFs | `srt/model_loader/expert_pack_runtime.py:472` | defaults are needed for two `generation_config` keys |

`ExpertPackStore` caches experts in VRAM and streams the rest from the pack
(`srt/layers/moe/expert_pack.py`), which is the partial-residency mechanism this
engine has. It is written for **V4-Flash MXFP4** GGUFs; the converter that
produces them from a safetensors V4.1 checkpoint is upstream llama.cpp's
`conversion/deepseek41.py`, whose header states it copies routed experts
bit-exact as MXFP4 and dequantises the fp8 linears to BF16.

### SGLang safetensors — V4.1 is supported, residency is not

The engine does branch on `model_type == "deepseek_v41"`:
`srt/configs/deepseek_v41.py`, `srt/layers/attention/deepseek_v4_backend.py:1285`
(`is_dsv41`), `srt/layers/attention/graph_variants.py:136`, and
`srt/layers/moe/mega_gate.py`. So "the engine cannot run this model" is wrong;
three things are missing for our artifact, in order of how fast they fail:

1. `config.json` is renamed to `BluewhaleForCausalLM` / `bluewhale` /
   `deepseek_v41_*`; restoring those four values loads it as
   `DeepseekV41Config` (40 layers, 5120 hidden, 384 experts, top-6).
2. Our `fp8_dense` requires `weight_block_size == [128, 128]`
   (`sglang_restomod/quant/sm70_fp8_dense/_method.py:390`, `:416`), and this
   checkpoint's dense weights are `[32, 32]`.
3. There is no offloading: `srt/models/deepseek_v4.py:3818` builds its layers
   without `offloader_kwargs`, while `srt/models/deepseek_v2.py:2973` passes
   them, and the offloader's granularity is whole expert tensors of a layer, not
   a subset of experts. Without it every expert must be resident, and 288.78 GB
   of MXFP4 experts does not fit four 32 GB cards under any measured per-card
   overhead (see [the overhead record](runs/2026-11-16-per-card-overhead.md)).

### SGLang plain GGUF — not a fallback

`gguf.MODEL_ARCH_NAMES` in the installed package has deepseek, deepseek2 and
deepseek2-ocr only, so `srt/model_loader/loader.py:3078` raises
`Unknown gguf model_type: deepseek_v4`, and `GGUF_HF_NAME_MAP_BUILDERS` has no
builder for it.

## The three errors, reproduced

Each is a schema key, not a name, and each stops a load before any weight is
read:

```
ValueError: The checkpoint you are trying to load has model type `bluewhale`
  but Transformers does not recognize this architecture.
ValueError: expected GGUF architecture deepseek4, got 'deepseek41'
ValueError: GGUF metadata is missing required field: general.sampling.temp
```

## Rules that follow

- **Never describe a V4-Flash run as bluewhale.** They are different models with
  different geometry, and a result from one does not transfer to the other.
- **Schema keys are engine input.** `architectures`, `model_type`,
  `general.architecture` and the `<arch>.*` prefix decide whether an engine can
  read an artifact at all. Restoring them where an engine needs them is not a
  naming loss; the rename rule covers the names we author, not the schema a
  consumer looks up.
- **A pack is an identity copy.** Building one from a non-MXFP4 set writes the
  bytes faithfully and the store then rejects every entry, so check the dtype
  gate before spending the disk.
- **Price capacity against the artifact the engine can read.** A residency
  fraction computed for a checkpoint the loader refuses describes nothing.
