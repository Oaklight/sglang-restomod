# AGENTS.md

Instructions for coding agents working in this repository.

## What this repository is

`sglang-restomod` is a **lightweight downstream layer** for SGLang. `main` must
stay light: it contains the plugin, the patch queue, recipes, and scripts, but
**not** the SGLang source tree.

- Upstream baseline: `UPSTREAM_PIN.toml`.
- `upstream` branch: pristine reference tree; never merge it into `main`.
- SGLang is fetched into `.build/sglang` only when a patch exists.

Read first: `README.md`, `RESTMOD.md`, `BRANCHES.md`, `patches/README.md`.

## Golden rules

1. Never vendor the SGLang tree or copy a whole downstream project.
2. Additive code goes in `sglang_restomod/`; register through
   `sglang.srt.plugins`.
3. An upstream edit requires a patch plus a `patches/manifest.json` entry that
   lists every touched path and a one-line reason.
4. Patch status is one of `pending`, `accepted`, `local-only`, `rejected`.
5. Do not delete a patch when an upstream PR merges. Retire it on the pin bump
   that contains the change, using `scripts/retire_patches.py`
   (reverse-apply check).
6. Do not add upstream `.github/workflows`; only `restomod-*.yml` belong here.
7. Do not store secrets. PyPI publishing uses Trusted Publishing (OIDC).

## Checks before proposing a change

```bash
cmp README.md README_en.md
python scripts/check_pin.py
python scripts/check_patches.py --skip-apply
# when patches exist:
scripts/fetch_upstream.sh
python scripts/check_patches.py
scripts/apply_patches.sh
python -m compileall -q sglang_restomod scripts
python -m build && twine check dist/*
```

## Versioning and README

- `__version__` in `sglang_restomod/__init__.py` is the only version source.
  `pyproject.toml` reads it with `[tool.setuptools.dynamic]`; never add a static
  version.
- `README.md` and `README_en.md` must stay byte-identical; CI checks with
  `cmp`. `README_zh.md` is the Chinese translation.

## Component selection

For a new component (for example a V100 attention backend):

- name the concrete need, target model, and hardware;
- record the source repository, commit, and license;
- select the minimal files and registration points; do not import an overlay;
- add a test or a recipe;
- update `COMPATIBILITY.md` only after a recorded run.

## Branches and PRs

- `main`: the layer; protected by PR plus `restomod-ci` and
  `restomod-patch-guard`.
- `upstream`: reference pin; only `restomod-pin-sync` moves it.
- `bump/upstream-*`: pin bumps opened by CI.
- Sign off commits (`git commit -s`) when DCO applies.

## Code style

Python 3.12, ruff, line length 100. Keep modules import-light: the plugin loads
at server startup.
