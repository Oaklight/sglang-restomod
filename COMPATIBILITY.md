# Compatibility matrix

Status is deliberately conservative: "planned" means the lane exists but has
not been measured from this repository.

| Hardware | Architecture | Engine lane | Status |
|---|---|---|---|
| Tesla V100-SXM2 | sm70 | this repository (`sglang-restomod`, v100_plus baseline) | Baseline imported; re-validate on lambda12 |
| Tesla T4 | sm75 | upstream SGLang | Planned |
| RTX 4070 Super | sm89 | upstream SGLang | Planned |
| H100 | sm90 | upstream SGLang | Planned |
| DGX Spark | sm121 / aarch64 | upstream SGLang | Planned |

Notes:

- Upstream SGLang supports sm75 and newer; it does not target sm70
  (sgl-project/sglang#931). V100 therefore runs this downstream, not stock
  SGLang.
- The V100 baseline targets 8x V100-SXM2 in two NVLink quads. Smaller TP/PP
  topologies need explicit validation; do not assume the 8-card profile scales
  down.
- Update this table only with a recorded run: hardware, engine pin, model,
  topology, context, and date.
