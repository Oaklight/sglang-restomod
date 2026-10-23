# Patches

Upstream-file changes live here. The baseline has **no patches**: every
component starts as additive code under `sglang_restomod/` unless a patch is
unavoidable.

## Adding a patch

1. Fetch the pinned source:

   ```bash
   scripts/fetch_upstream.sh
   ```

2. Edit `.build/sglang` directly, then export the diff:

   ```bash
   git -C .build/sglang diff > patches/0001-short-name.patch
   git -C .build/sglang checkout -- .
   ```

3. Record it in `patches/manifest.json`:

   ```json
   {
     "upstream": "baa9befa381511cd4d56f83980951e5a45fffc6d",
     "patches": {
       "0001-short-name.patch": {
         "paths": ["python/sglang/srt/layers/attention/..."],
         "reason": "Register the selected SM70 attention backend.",
         "source": "owner/repo@commit",
         "upstream_status": "pending"
       }
     }
   }
   ```

4. Verify:

   ```bash
   scripts/apply_patches.sh
   python scripts/check_patches.py
   ```

## Rules

- One logical change per patch; keep the touched paths minimal.
- Every path in the patch must be listed in the manifest.
- Prefer a registration-point edit over rewriting upstream logic.
- If upstream accepts the change, delete the patch and the manifest entry.
- Name patches `NNNN-short-name.patch`; order is applied lexicographically.

## Retiring a patch after upstream accepts it

Do **not** delete a patch when the upstream PR merges: the pinned source may not
contain the change yet, and `main` would silently lose the fix. Retire it on the
pin bump that first contains the upstream change.

1. When the upstream PR merges, set `upstream_status` to `accepted` and record
   `upstream_pr` and, if known, `accepted_in`.
2. On the next pin bump, `scripts/retire_patches.py` checks each accepted patch
   against the new source with `git apply --check --reverse`. If the reverse
   check succeeds, upstream already contains the change: delete the patch file
   and its manifest entry in the bump PR.
3. If the reverse check fails, the patch is still needed (or conflicts). Keep
   it, rebase it, and leave the status as `accepted` until a later pin contains
   the change.

`restomod-patch-guard` fails when an accepted patch is still present and already
reverse-applies on the pin, so a missed retirement cannot sit silently.

Status values: `pending`, `accepted`, `local-only`, `rejected`.

