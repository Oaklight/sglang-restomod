# External code: where it comes from and how a fix reaches a launch

This layer consumes code from three kinds of source, and each kind has a
different sync path. A change that is correct in a source repository is inert
here until the path below is walked. This page is the map, and it records what
is currently stranded on either side.

## The three paths

| Source | Kind | Pinned in | Copy in this repository | Check |
|---|---|---|---|---|
| [`Oaklight/sm70-kernels`](https://github.com/Oaklight/sm70-kernels) | git submodule, source only | `UPSTREAM_PIN.toml` `[kernels]` | `third_party/sm70-kernels/` (submodule), then two generated files | `scripts/sync_kernels.py --check` |
| [`Oaklight/lsa-sm70`](https://github.com/Oaklight/lsa-sm70) | installed package | `UPSTREAM_PIN.toml` `[lsa]` | none; the `lsa_sm70` component is a registration shim | `scripts/check_pin.py` (version only) |
| `marlin_v100`, TurboMind, 1Cat | prebuilt `.so` | not in this repository | none; located at run time | none automatic |

### sm70-kernels: submodule, generated copy

`third_party/sm70-kernels` is a submodule at the commit in `[kernels].commit`.
Two files are copied into the package by `scripts/sync_kernels.py`, and their
SHA256 values are recorded in the pin:

| source | destination |
|---|---|
| `sm70_kernels/longctx_decode/csrc/sm70_longctx_decode.cu` | `sglang_restomod/attention/v100/kernels/sm70_longctx_decode.cu` |
| `sm70_kernels/tilelang_fa/_kernels_paged.py` | `sglang_restomod/attention/v100/tilelang/_kernels_paged.py` |

The destinations are **generated, not edited**. To change one, change
sm70-kernels, bump `[kernels].commit`, then run the sync. A local edit to a
destination is overwritten on the next sync and fails `--check` before that.

```sh
git submodule update --init --recursive
python scripts/sync_kernels.py --check   # non-zero on any drift
```

The rest of sm70-kernels (its loader, tests, benchmarks, docs) is not vendored
and has no sync path. `_loader.py` and this layer's `tilelang/_decode_cuda.py`
are separate implementations that have already diverged; see "Stranded" below.

### lsa-sm70: package, version-checked

The `lsa_sm70` component imports the installed package; no LSA implementation
lives here. The pin records both a commit and a version. `check_pin.py` compares
the installed `__version__` only, so a commit that does not change the version
is not detected.

### Prebuilt binaries: no sync path at all

The SM70 FP8 kernels are loaded from `_sm70_turbomind_v100.so` by path
(`SM70_TURBOMIND_SO`, `SGLANG_SM70_TURBOMIND_OPS_PATH`), and the NVFP4/WNA16
Marlin extension from `SM70_MARLIN_DIR`. Both are built outside this repository
(`docs/quantization-dispatch.md`, each component's `PROVENANCE.md`).

When a fix lands in the upstream source of one of those libraries, the binary
we load still has the old behaviour. There are two honest responses:

1. **Port the invariant to Python** when the fix is validation, routing or a
   shape guard. This is what #264 did: the upstream shape check was
   reimplemented in Python because the prebuilt symbol could not change.
2. **Rebuild the binary** at or after the fix commit. That is a site/infra
   step, not a patch in this repository, and it changes what every launch
   loads.

Do not describe a prebuilt-binary fix as shipped until the rebuild has happened
and a recorded run names the new build.

## What "inert" looks like

- A kernel fix merged in sm70-kernels but not pinned and synced here: the
  package still carries the old source, and `sync_kernels.py --check` is green
  because the pin and the copy agree with each other.
- A fix in the upstream source of a prebuilt `.so`: nothing in this repository
  changes, and no check can see it.
- A fix made directly in a generated destination: the next sync reverts it.
- A `lsa-sm70` commit with no version bump: `check_pin.py` stays green.

The common shape is that the check compares this repository with its own pin,
not with the source repository's `main`. A green check means "consistent with
the pin", never "current with upstream".

## Stranded on 2026-11-15

Facts verified against the trees, not inferred:

| Side | Item | Evidence |
|---|---|---|
| this repository | `tilelang/_decode_cuda.py` raises `NotImplementedError` for an E4M3 KV cache and says the Python side never selects that specialisation. The synced `.cu` does dispatch E4M3 (`dispatch_serial` selects `launch_serial<uint8_t, kCtas, true>`), and sm70-kernels' own `_loader.py` accepts it. | `_decode_cuda.py` guard vs `sm70_longctx_decode.cu` `a.e4m3` branches |
| this repository | `PROVENANCE.md` says the `.cu` "carries `sm70-decode-occ2.patch`". No such patch exists; the file is generated from the pin, where occ2 is part of the source history. | `patches/` holds only the three upstream patches |
| this repository | `scripts/sync_kernels.py --check` is not in CI. `pyproject.toml` says the submodule repository is private; it is public, so the check could run. | `restomod-ci.yml` runs `check_pin.py`, not `sync_kernels.py` |
| sm70-kernels | the pin `bdf1e56` is a feature-branch commit; `origin/main` contains the same tree at the merge `c393136`. Re-pin to the main-line commit so the pin is reachable from the branch the project treats as stable. | `git diff --stat bdf1e56 origin/main` is empty |
| sm70-kernels | no `LICENSE` or `NOTICE` file at the pin, although `pyproject.toml` declares Apache-2.0. The fix is on an unmerged `legal/license-notice` branch. This repository redistributes two derivative files and lists neither sm70-kernels nor its GooseLLM lineage in `THIRD_PARTY.md`. | tree listing; `THIRD_PARTY.md` |
| lsa-sm70 | the pin is one commit behind `main`; the delta is a benchmark script and the version is unchanged. | `git rev-list --count 3a47980..ea9264a` is 1 |

The first item is the highest-value one: the shipped kernel already supports
E4M3 KV, and a stale guard in this layer refuses it before the call.
