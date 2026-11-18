# Provenance: paged_logits_sm70

- **Origin**: authored in this repository for the SM70 target. No third-party
  code is vendored; the semantics are the pin's own torch twin,
  `fp8_paged_mqa_logits_torch` (`sglang/srt/layers/attention/dsv4/indexer.py:82`),
  reproduced term for term: `relu` before the head weighting, the per-token
  fp32 scale applied once after the relu-weighted head sum, and the `bmm`
  result rounded to bf16 first (which is what a bf16 `torch.bmm` returns and
  what the top-64 check sees).
- **Kernel**: Triton 3.5.1 source in `kernel.py`, written here. No external
  repository, commit, or license applies. The e4m3 decode is integer arithmetic
  because Triton rejects the `fp8e4nv` dtype on this architecture.
- **Selected files**: `sglang_restomod/kernels/paged_logits_sm70/__init__.py`,
  `kernel.py`. Registration point: the shim's `install_fused_paged_mqa_logits()`,
  reached from `register()`; the dispatch switch (`SGLANG_SM70_PAGED_LOGITS`) and
  the B <= 4 threshold are the shim's, not this component's.
- **Hardware**: measured on one Tesla V100-SXM2-32GB (SM70, cap 7.0), CUDA 12.8,
  torch 2.9.1+cu128, Triton 3.5.1. Evidence:
  `docs/runs/2026-11-17-indexer-kernel-first-run.md`.
- **Status**: `local-only`, small-batch arm only. Captured, replay-only:
  3.51x at B=1, 2.48x at B=2, 2.25x at B=4, ~1.95x from B=8 up, with 1.2e-7
  agreement and an identical top-64 set at every batch. The B=32 gate (>= 3x)
  is **not met**; the dot has no tensor-core lowering under Triton on sm70
  (fp16 lands within noise of fp32, bf16 is 10x slower), so the large-batch arm
  needs a TileLang rewrite.
- **Not A/B'd in an engine**: no served step measurement exists for this op on
  this hardware. Every ratio here, and the ~1.2 ms/step worth at B <= 4, is a
  single-op inference from a synthetic call.
