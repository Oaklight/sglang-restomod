# Third-party provenance

At baseline this repository contains **no code from the V100 projects**; they
are candidate sources for selected components.

| Component | Source | License | Status |
|---|---|---|---|
| SGLang | `sgl-project/sglang` | Apache-2.0 | Pinned dependency; not vendored |
| sglang-v100-plus | `heislera763/sglang-v100-plus` | Apache-2.0 | Candidate source; nothing imported yet |
| sglang-V100 | `haohervchb/sglang-V100` | Apache-2.0 (verify) | Candidate source |
| sglang-sxm2 | `dg1kjd/sglang-sxm2` | Apache-2.0 (verify) | Candidate source |
| marlin-v100 | `heislera763/marlin-v100` | see upstream | Candidate kernel dependency |
| flash-attention-v100 | `heislera763/flash-attention-v100` | see upstream | Candidate kernel dependency |

Rules:

- Every selected component records its source repository, commit, and license
  in the PR and in the file header or manifest entry.
- Retain the upstream `LICENSE`/`NOTICE` when redistributing.
- This file is informational, not legal advice.
