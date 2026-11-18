# Provenance: hc_prenorm_sm70

- **Origin**: authored in this repository for the SM70 target. No third-party
  code is vendored; the math is the pin's own `hc_pre_torch_impl`
  (`deepseek_v4.py:3358`), restated as `_hc_prenorm_reference` in
  `sglang_restomod/kernels/deep_gemm_sm70/__init__.py`.
- **Kernel**: Triton 3.5.1 source in `kernel.py`, written here. No external
  repository, commit, or license applies.
- **Selected files**: `sglang_restomod/kernels/hc_prenorm_sm70/__init__.py`,
  `kernel.py`. Registration point: the shim's `install_fused_hc_prenorm()`,
  reached from `register()`; the dispatch switch (`SGLANG_SM70_HC_PRENORM`) is
  the shim's, not this component's.
- **Hardware**: measured on one Tesla V100-SXM2-32GB (SM70, cap 7.0), CUDA 12.8,
  torch 2.9.1+cu128. Evidence: `docs/runs/2026-11-17-prenorm-kernel-gate.md`.
- **Status**: `local-only`. It is the large-M arm only. Decode shapes lose by
  4-7x inside a captured graph, so it must not be the default arm.
- **Not yet A/B'd in an engine.** `hc_pre_torch_impl` lives in
  `deepseek_v4.py`, and no DeepSeek-V4 forward exists on this hardware yet
  (Flash-Next's config has no `hc_mult`, so it cannot exercise the call). Every
  ratio in this component is a single-op measurement, not a serving number.
  When a bluewhale serve exists, this kernel and the indexer should be A/B'd in
  one window, on the same prompts, with the switch arms declared.

- **Receipt note (2026-11-17)**: a PR touching `sglang_restomod/kernels/` needs a
  GPU receipt, but the policy job in `.github/workflows/restomod-gpu-receipt.yml`
  reports `requires=false` for it — the path alternation at line 106 lists
  `(quant|attention|moe|models|spec|linear_attn_sm70)` and omits `kernels/`.
  Until that list is fixed, a green `gpu-receipt` is **not** evidence for this
  component; treat the receipt as required.
