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

Baseline only: `sglang==0.5.22`, an empty patch queue, and a no-op plugin. V100
support will arrive as selected components; until then this layer does nothing
on its own.

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
