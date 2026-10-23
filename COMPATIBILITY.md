# Compatibility matrix

The baseline layer is empty: it pins SGLang and ships no V100 components. V100
support is added component by component, and a row becomes "validated" only
after a recorded run.

| Hardware | Architecture | Path | Status |
|---|---|---|---|
| Tesla V100-SXM2 | sm70 | this layer + selected components | Baseline only; no components selected |
| Tesla T4 | sm75 | upstream SGLang 0.5.22 | Planned |
| RTX 4070 Super | sm89 | upstream SGLang 0.5.22 | Planned |
| H100 | sm90 | upstream SGLang 0.5.22 | Planned |
| DGX Spark | sm121 / aarch64 | upstream SGLang 0.5.22 | Planned |

Notes:

- Upstream SGLang supports sm75 and newer; it does not target sm70. V100 needs
  selected components from this layer.
- A validated row records: hardware, upstream pin, component versions, model,
  topology, context, date.
