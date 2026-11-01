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

Two components ship, both opt-in via `SGLANG_RESTOMOD_COMPONENTS`:

| component | what it does |
|---|---|
| `fp8_moe` | block-FP8 expert GEMM on sm70. The TurboMind library SGLang ships for SM70 exports `fp8_moe_gemm_sm70_out` but never registers it with torch; this resolves the symbol with a small `dlopen`/`dlsym` shim and exposes a typed surface. This is what makes the official block-FP8 MoE checkpoints runnable on V100. |
| `attention_v100` | implements `init_forward_metadata_out_graph` on the V100 attention backend, which the graph runner calls and the base class leaves as a no-op. Without it a decode CUDA graph cannot capture. |

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
