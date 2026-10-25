#!/usr/bin/env bash
# Apply the repository rulesets. Requires a public repository or GitHub Pro.
#
# Usage: scripts/apply_rulesets.sh [owner/repo]
set -euo pipefail

repo="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
api="repos/${repo}/rulesets"

if ! gh api "repos/${repo}/rulesets" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: rulesets are unavailable for this repository.
GitHub Free private repositories do not support branch protection or rulesets.
Make the repository public (or upgrade the account to Pro), then rerun.
EOF
  exit 1
fi

upsert() {
  local name="$1"
  local payload="$2"
  local id
  id="$(gh api "${api}" --jq ".[] | select(.name == \"${name}\") | .id" 2>/dev/null || true)"
  if [[ -n "$id" ]]; then
    printf '%s' "$payload" | gh api -X PUT "${api}/${id}" --input -
  else
    printf '%s' "$payload" | gh api -X POST "${api}" --input -
  fi
}

upsert "main-protection" '{
  "name": "main-protection",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["refs/heads/main"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "required_status_checks": [
          { "context": "restomod-ci" },
          { "context": "restomod-patch-guard" },
          { "context": "gpu-receipt" }
        ],
        "strict_required_status_checks_policy": true
      }
    }
  ],
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" }
  ]
}'

# Personal repositories cannot use the GitHub Actions integration as a ruleset
# bypass actor (API 422). Keep `upstream` deletion-protected; bot-only
# force-push protection would need a write deploy key added as a DeployKey
# bypass actor. See BRANCHES.md for the upgrade path.
upsert "upstream-pin" '{
  "name": "upstream-pin",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["refs/heads/upstream"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" }
  ],
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" }
  ]
}'

upsert "release-tags" '{
  "name": "release-tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["refs/tags/restomod-v*"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "update" }
  ],
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" }
  ]
}'

upsert "upstream-tags" '{
  "name": "upstream-tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["refs/tags/upstream-v*"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" }
  ],
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" }
  ]
}'

echo "applied rulesets for ${repo}"
