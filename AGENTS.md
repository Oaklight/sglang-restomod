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

## This is a mod, not a mirror

The project exists because SGLang does not work well on SM70. Where upstream is
wrong for this hardware — a kernel that lowers `tl.dot` to scalar FMA, a backend
that assumes SM80, a shape gate that refuses a Volta-only layout, a loader that
drops the engine when it is on `PYTHONPATH` — **we change it**. That is the
project, not an escalation, and it does not need anyone's permission.

The patch queue and the component interface are the bookkeeping for those
changes, not a gate on whether to make them. If the right fix is an upstream
edit, write the patch and the manifest entry; if it is a component that replaces
a path at runtime, write the component; if it is a new kernel, write the kernel.
Do not settle for a small test or a workaround because the large fix needs a
patch entry, and do not describe a limitation as a design choice when it is only
an unexamined default.

What still binds, and is not timidity: nothing lands without evidence that it
works on the hardware, and nothing is called verified until a run says so. The
confidence to change upstream is matched by the obligation to measure what the
change did.

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

## Components

A component is a module under `sglang_restomod/` with a `register()` entry
point, selected by name:

```sh
SGLANG_RESTOMOD_COMPONENTS=fp8_moe,attention_v100 SGLANG_PLUGINS=restomod ...
```

Rules:

1. Import is cheap. No torch, no sglang, no kernel loading at module import;
   `register()` must not do heavy work either. Load on first use.
2. Every component ships `PROVENANCE.md` (repo, commit, license, selected
   files), a CPU-runnable unit test, and a GPU-marked test when it needs
   hardware.
3. GPU tests carry `@pytest.mark.gpu` and are excluded from `restomod-ci`;
   run them with `scripts/test_gpu.sh` on the GPU host.
4. Unset or empty `SGLANG_RESTOMOD_COMPONENTS` means baseline: the plugin
   loads and does nothing.

## Branches and PRs

- `main`: the layer; protected by PR plus `restomod-ci` and
  `restomod-patch-guard`.
- `upstream`: reference pin; moved only by `scripts/move_upstream.sh`, and
  `restomod-pin-sync` fails when it is stale.
- `bump/upstream-*`: pin bumps opened by CI.
- Sign off commits (`git commit -s`) when DCO applies.

## Code style

Python 3.12, ruff, line length 100. Keep modules import-light: the plugin loads
at server startup.
