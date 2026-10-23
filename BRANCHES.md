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

`upstream` is a read-only pin in practice; `restomod-pin-sync` is its only
writer.

- deletion is blocked;
- force-push/update restriction is **not** applied on this personal repository:
  GitHub does not allow the Actions integration as a ruleset bypass actor
  (HTTP 422). Until a deploy-key bypass is configured, treat "only pin-sync
  moves `upstream`" as a convention, not an enforced rule.

Upgrade path for enforced bot-only force-push protection:

1. create an ed25519 deploy key with write access and add it to the repository;
2. store the private key as `PIN_SYNC_SSH_KEY`;
3. configure `restomod-pin-sync` to push over SSH with that key;
4. add the deploy key as a `DeployKey` bypass actor and re-add
   `non_fast_forward` + `update` to the `upstream-pin` ruleset.

### Tags

- `restomod-v*`: immutable (no update, no deletion); releases are cut from
  `main`.
- `upstream-v*`: deletion-protected; `restomod-pin-sync` may retag it. Update
  protection is omitted for the same personal-repo reason as `upstream`.

## Bypass actors

| Actor | ID | Type |
|---|---|---|
| Repository admin role | 5 | `RepositoryRole` |

The GitHub Actions integration (15368) is not a usable bypass actor on a
personal repository; it is mentioned only as the intended actor for the
deploy-key upgrade.

## Verifying after applying

1. `gh api repos/$OWNER/$REPO/rulesets` lists four active rulesets:
   `main-protection`, `upstream-pin`, `release-tags`, `upstream-tags`.
2. Open a PR: `main` requires `restomod-ci` and `restomod-patch-guard`.
3. Run `restomod-pin-sync` manually: it must still move `upstream` and retag
   `upstream-v*`.
4. A human force-push to `upstream` is currently allowed by convention; after
   the deploy-key upgrade it must be rejected.
