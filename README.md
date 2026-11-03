<p align="center">
  <a href="README_en.md">English</a> | <a href="README_zh.md">中文</a>
</p>

# sglang-restomod

[![PyPI version](https://img.shields.io/pypi/v/sglang-restomod?color=green)](https://pypi.org/project/sglang-restomod/)
[![CI](https://github.com/Oaklight/sglang-restomod/actions/workflows/restomod-ci.yml/badge.svg)](https://github.com/Oaklight/sglang-restomod/actions/workflows/restomod-ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

A **lightweight downstream layer** for [SGLang](https://github.com/sgl-project/sglang).

`main` does not contain the SGLang source tree. It contains only:

- a pinned upstream baseline (`UPSTREAM_PIN.toml`);
- a plugin package (`sglang_restomod/`);
- a small, auditable patch queue (`patches/`);
- recipes and CI.

Upstream components are added deliberately, one at a time, with provenance and
tests. The repository is not a fork of SGLang or of any V100 project.

## Status

Seven components ship, all opt-in via `SGLANG_RESTOMOD_COMPONENTS`:

| component | what it does |
|---|---|
| `fp8_moe` | block-FP8 expert GEMM on sm70. The TurboMind library SGLang ships for SM70 exports `fp8_moe_gemm_sm70_out` but never registers it with torch; this resolves the symbol with a small `dlopen`/`dlsym` shim and exposes a typed surface. This is what makes the official block-FP8 MoE checkpoints runnable on V100. |
| `fp8_dense` | SM70 W8A16 dense FP8 linear kernels for the pinned engine. |
| `attention_v100` | V100 full attention for the pin. Vendors `flash_attn_v100_backend.py`, the SM70 QSA backend and the `tilelang_fa_v100` kernels, registers the `tilelang_fa_v100` backend name (the pin does not define it), and implements `init_forward_metadata_out_graph` on the backend, which the graph runner calls and the base class leaves as a no-op. Without the last one a decode CUDA graph cannot capture; without the first two the launch does not start. |
| `fp4_moe` | NVFP4 expert GEMM through the `marlin_v100` extension: load-time repack, SM70 scale encoding, and the GEMM bridge. The Marlin kernel SGLang ships is an empty body below SM80. |
| `ple_sm70` | SM70 decode for the fp8 PLE host-embedding gather, which the pinned kernel cannot compile because it casts to `tl.float8e4nv`. |
| `spec_sm70` | top-k/top-p renorm replacements so non-greedy EAGLE verify does not reach FlashInfer's sm75 floor. |
| `gdn_tilelang_sm70` | SM70 TileLang GDN chunked prefill. The pin's Triton chunk kernels take 63.9% of a 27B prefill chunk; this component answers the same call with a TileLang kernel: 1,933.6 / 1,924.9 tok/s at 8,041 prompt tokens and 1,658.3 / 1,655.1 at 31,158, against 719.9 and 684.6 on Triton (receipt `evidence-55cb6e7-gdn-tilelang`). |

```bash
SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100
```

With the variable unset the plugin stays inert. The patch queue is still empty
(`patches/manifest.json`), so nothing here edits upstream files; it all arrives
through the plugin entry point.

## Reference projects

These projects define the behaviour this layer reproduces. Only the code named
in a component's `PROVENANCE.md` is used; the rest is reference material.

| Project | Used for |
|---|---|
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Upstream engine at the pin. |
| [heislera763/sglang-v100-plus](https://github.com/heislera763/sglang-v100-plus) | V100 baseline and integration reference; local snapshot at `sglang-restomod-reference`. |
| [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) | Original Volta port in that lineage. |
| [dg1kjd/sglang-sxm2](https://github.com/dg1kjd/sglang-sxm2) | `attention_v100` backend and the long-context decode kernel. |
| [heislera763/flash-attention-v100](https://github.com/heislera763/flash-attention-v100) | Dense prefill attention used by `attention_v100`. |
| [zhinianqin/marlin_v100](https://github.com/zhinianqin/marlin_v100) | SM70 Marlin kernels for block-FP8 and NVFP4 expert GEMM. |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | FP8 expert GEMM contract, and the published Flash-Next number this repo compares against. |
| TurboMind ([LMDeploy](https://github.com/InternLM/lmdeploy)) | Prebuilt SM70 block-FP8 expert GEMM bound by `fp8_moe`. |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Build dependency of the SM70 Marlin extension (v4.2.1). |

## Performance

Qwen3.8-Flash-Next and Qwen3.8-27B on V100-SXM2-32GB, one node, decode
throughput in tok/s at 128 output tokens from `scripts/bench.py`. A row is only
meaningful with its model, precision, card count, and sampling mode; nothing
here is comparable across those axes.

| Model | Precision | Cards | Parallel | MTP | Sampling | Prompt | Decode | Source |
|---|---|---|---|---|---|---|---|---|
| Flash-Next | FP8 | 8 | TP8 / EP8 | off | default | 76 | 40.0 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | FP8 | 8 | TP8 / EP8 | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 73.9 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | FP8 | 8 | TP8 / EP8 | 3/4 | temperature 0 | 74 | 98.3—106.2 | `recipes/flash-next-fp8-v100-tp8ep8-mtp.md` |
| Flash-Next | NVFP4 | 4 | TP4, no EP | 3/4 | temperature 0 | 993 | 87.27 / 87.24 | receipt `evidence-6294562-nvfp4` |
| Flash-Next | NVFP4 | 4 | TP4, no EP | 3/4 | temperature 0 | 3473 | 72.43 / 90.68 | receipt `evidence-6294562-nvfp4` |
| Flash-Next | NVFP4 | 4 | TP4, no EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 74 | 81.83 / 88.06 | receipt `evidence-6294562-nvfp4` |
| 27B | FP8 | 2 | TP2, no EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 8037 | 69.30 / 63.24 | receipt `evidence-e5a5740-draft-extend` |
| 27B | FP8 | 2 | TP2, no EP | 3/4 | temp 1, top-p 0.95, top-k 20 | 31109 | 44.59 / 45.79 | receipt `evidence-e5a5740-draft-extend` |
| 27B | FP8 | 2 | TP2, no EP, 40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 8037 | 73.33 / 66.43 | receipt `evidence-153a4c2-kv-splits` |
| 27B | FP8 | 2 | TP2, no EP, 40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 31158 | 54.42 / 47.13 | receipt `evidence-153a4c2-kv-splits` |
| 27B | FP8, e5m2 KV | 2 | TP2, no EP, 40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 8030 | 67.11 / 65.54 | receipt `evidence-7e15a2b-fp8-draft-extend` |
| 27B | FP8, e5m2 KV | 2 | TP2, no EP, 40 KV splits | 3/4 | temp 1, top-p 0.95, top-k 20 | 32112 | 54.39 / 46.69 | receipt `evidence-7e15a2b-fp8-draft-extend` |

The 27B rows need `SGLANG_V100_DRAFT_EXTEND_NATIVE=1` and
`--speculative-draft-attention-backend tilelang_fa_v100`. Without them the same
box measured 17.9 tok/s at 31.7k (issue #75).

The 40-split rows are a separate configuration: they add
`--triton-attention-num-kv-splits 40`, which raises the draft decode kernel from
16 CTAs to 80. The recipe says what it costs in log-probability; do not compare
those rows against the default ones as if only the code changed.

The e5m2 rows are the same launch with `--kv-cache-dtype fp8_e5m2`, which halves
the KV bytes. They need the draft-extend routing fix in #129: before it, the fp8
cache sent the draft extend to the Triton extend kernel and 32k decode fell to
36.4 tok/s. At 32k they measure the same as the fp16 rows; at 8k they are 5-10%
lower. Prefill does not move.

How to read it:

- Decode throughput is `accepted tokens per step / step time`. A greedy row and
  a sampled row differ in acceptance, so they must not be compared.
- Rows from different days differ by host state as much as by code. The same
  box measured 87.27 tok/s for `6294562` at 16:00 and 70.32 tok/s for `303e162`
  an hour later. A back-to-back A/B of both heads under identical conditions
  (pull request #96) shows up to 10% between commits, so the rest of that
  19 tok/s spread is host state, and it is unexplained.
- FP8 rows use 8 cards, NVFP4 rows use 4. Per-card throughput is the
  comparable quantity, not the row total.
- Update this table when a new signed receipt lands, and only with numbers that
  cite a receipt or a recipe. A row without a citation does not belong here.

Which of those configurations is validated, and which are still planned, is
[`COMPATIBILITY.md`](COMPATIBILITY.md): one row per hardware and stack, each
carrying its receipt or its blocking issue.

## Quick start

```bash
pip install sglang-restomod
SGLANG_PLUGINS=restomod python -m sglang.launch_server --help
```

For development:

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -e . --no-deps
```

With patches present, build from the pinned source instead:

```bash
scripts/build.sh
```

## Layout

```
pyproject.toml          sglang-restomod; pins sglang==0.5.22
sglang_restomod/        plugin, overrides, selected components
patches/                upstream-file changes + manifest
recipes/                validated launch profiles
scripts/                fetch, patch, build, check, bump
UPSTREAM_PIN.toml       the baseline
```

## Branches

| Branch | Meaning |
|---|---|
| `main` | the lightweight layer |
| `upstream` | pristine SGLang tree at the pin; reference only, never merged |
| `bump/upstream-*` | proposed pin bumps, opened by CI |

See [RESTMOD.md](RESTMOD.md) for the component-selection and patch-retirement
rules, and [patches/README.md](patches/README.md) for the patch lifecycle.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Coding-agent instructions are in
[AGENTS.md](AGENTS.md).

## License

Apache-2.0. SGLang and any selected component keep their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md).
