# Branch and tag rules

## Current state

The repository is **public**, so the rulesets under `main-protection`,
`upstream-pin`, `release-tags` and `upstream-tags` are active. `main` requires a
PR plus `restomod-ci`, `restomod-patch-guard` and `gpu-receipt`, and requires
the head to be up to date with `main`.

- never push directly to `main`; use a PR;
- move `upstream` only with `scripts/move_upstream.sh`, from a maintainer
  checkout whose credential may push workflow files;
- never force-push `main`.

`scripts/apply_rulesets.sh` created the rulesets.

## Intended rules

### `main`

- require a pull request (0 approvals, so a solo maintainer can merge);
- require status checks `restomod-ci`, `restomod-patch-guard` and
  `gpu-receipt`;
- require the branch to be up to date before merging;
- block deletion and non-fast-forward pushes;
- allow repository admins to bypass in an emergency.

### `upstream`

`upstream` holds the pristine SGLang tree at the pin. It is moved by
`scripts/move_upstream.sh` and checked by `restomod-pin-sync`.

- the move cannot run in CI: the tree carries 114 `.github/workflows/` files,
  and a `GITHUB_TOKEN` may not create or update a workflow file
  (`refusing to allow a GitHub App to create or update workflow ... without
  workflows permission`);
- deletion is blocked;
- force-push/update restriction is **not** applied on this personal repository:
  GitHub does not allow the Actions integration as a ruleset bypass actor
  (HTTP 422). Treat "only `move_upstream.sh` moves `upstream`" as a convention,
  not an enforced rule.

Upgrade path for enforced maintainer-only force-push protection:

1. create an ed25519 deploy key with write access and add it to the repository;
2. store the private key as a secret and push with it from `move_upstream.sh`;
3. add the deploy key as a `DeployKey` bypass actor and re-add
   `non_fast_forward` + `update` to the `upstream-pin` ruleset.

### Tags

- `restomod-v*`: immutable (no update, no deletion); releases are cut from
  `main`.
- `upstream-v*`: deletion-protected; `scripts/move_upstream.sh` may retag it.
  Update protection is omitted for the same personal-repo reason as `upstream`.

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
2. Open a PR: `main` requires `restomod-ci`, `restomod-patch-guard` and
   `gpu-receipt`.
3. Run `restomod-pin-sync` manually: `scripts/move_upstream.sh --check` must
   pass when the branch and the tag are at the pin, and fail with instructions
   when either is stale.
4. A human force-push to `upstream` is currently allowed by convention; after
   the deploy-key upgrade it must be rejected.
