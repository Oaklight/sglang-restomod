# Branch and tag rules

## Current state

The repository is **private on GitHub Free**, where branch protection and
rulesets are unavailable (`Upgrade to GitHub Pro or make this repository
public`). Until it is public or the account has Pro:

- never push directly to `main`; use a PR (the checks still run on push, but
  they do not block);
- never move `upstream` by hand; `restomod-pin-sync` is its only writer;
- never force-push `main`.

When the repository becomes public (or with Pro), run:

```bash
scripts/apply_rulesets.sh
```

## Intended rules

### `main`

- require a pull request (0 approvals, so a solo maintainer can merge);
- require status checks `restomod-ci` and `restomod-patch-guard`;
- require the branch to be up to date before merging;
- block deletion and non-fast-forward pushes;
- allow repository admins to bypass in an emergency.

### `upstream`

`upstream` is a read-only pin; only the pin-sync bot may move it.

- block deletion;
- block non-fast-forward pushes for humans;
- restrict updates to the GitHub Actions integration and repository admins
  (bypass actors), so `restomod-pin-sync` can force-update the branch;
- the branch always points at `UPSTREAM_PIN.toml`'s `commit`.

### Tags

- `restomod-v*`: immutable (no update, no deletion); releases are cut from
  `main`.
- `upstream-v*`: movable only by `restomod-pin-sync` (GitHub Actions
  integration bypass), so a pin bump can retag.

## Bypass actors

The ruleset script uses these actor IDs:

| Actor | ID | Type |
|---|---|---|
| Repository admin role | 5 | `RepositoryRole` |
| GitHub Actions | 15368 | `Integration` |

If a future GitHub change makes the Actions bypass ineffective, the pin-sync
workflow will fail loudly; switch it to a deploy key or a fine-grained PAT and
update the bypass list rather than removing the rule.

## Verifying after applying

1. `gh api repos/$OWNER/$REPO/rulesets` lists three active rulesets.
2. Open a trivial PR: `main` requires `restomod-ci` and
   `restomod-patch-guard`.
3. Run `restomod-pin-sync` manually: it must still be able to move `upstream`.
4. Try a force-push to `upstream` as a human: it must be rejected.
