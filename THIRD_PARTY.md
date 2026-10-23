# Third-party provenance

This repository is a downstream of SGLang and vendors or links several V100
projects. Keep this file current before any redistribution.

| Component | Source | License | How it is used |
|---|---|---|---|
| SGLang | `sgl-project/sglang` | Apache-2.0 | Upstream tree; pinned in `UPSTREAM_PIN.toml` |
| sglang-v100-plus | `heislera763/sglang-v100-plus` | Apache-2.0 | Vendored V100 baseline (`v100_plus/`, overlay, manifest) |
| sglang-V100 | `haohervchb/sglang-V100` | Apache-2.0 (verify) | Original Volta port; provenance recorded in `v100_plus/provenance.json` |
| sglang-sxm2 | `dg1kjd/sglang-sxm2` | Apache-2.0 (verify) | Reference for agentic-model V100 work; not a merge base |
| marlin-v100 | `heislera763/marlin-v100` | see upstream | SM70 Marlin kernels; pinned by `v100_plus/setup.sh` |
| flash-attention-v100 | `heislera763/flash-attention-v100` | see upstream | SM70 attention kernels; pinned by `v100_plus/setup.sh` |

Notes:

- The `v100_plus/provenance.json` file records the exact upstream base, kernel
  revisions, and content hashes for the vendored baseline. Do not edit it by
  hand; `scripts/upstream_bump.py` updates the upstream pin.
- Before publishing the repository or shipping binaries, re-check each
  upstream license and retain its `LICENSE`/`NOTICE` files.
- This file is informational, not legal advice.
