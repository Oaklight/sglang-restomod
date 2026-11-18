# Provenance: deep_gemm_sm70

- **Origin**: authored in this repository for the SM70 target. No third-party
  code is vendored. The entry names and signatures are DeepGEMM's public API as
  the pin calls it (`sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` and the
  call sites mapped in the SM70 DeepGEMM survey); the bodies are either the
  pin's own torch fallbacks or refusals.
- **Implemented**: `bf16_gemm_nt` (cuBLAS through torch), `fp8_paged_mqa_logits`
  (the pin's torch twin), `fp8_mqa_logits` (torch reference), and
  `tf32_hc_prenorm_gemm` (the pin's `hc_pre_torch_impl` math, with a
  `torch`/`fused`/`auto` switch whose fused arm is filled by
  `sglang_restomod.kernels.hc_prenorm_sm70`), plus the configuration surface
  (`set_pdl`, `set/get_num_sms`, `set/get_compile_mode`,
  `get_mk_alignment_for_contiguous_layout`).
- **Refused permanently**: `fp8_gemm_nt`, the grouped pair, `fp8_fp4_gemm_nt`.
  Kernel-level entries exist — `torch.ops.sglang_sm70_turbomind.fp8_gemm`,
  `torch.ops.sgl_kernel.moe_wna16_marlin`, `torch.ops._C.marlin_gemm` — but they
  take pre-packed W8A16 weights with per-layer leading dimensions, or routing
  metadata; forwarding would duplicate the quantisation.
- **Absent on purpose**: `fp8_fp4_mqa_logits`, `fp8_mega_moe`,
  `mega_moe_pre_dispatch_sm90`. The pin selects paths with `hasattr` on those
  names, so a stub would switch a path on.
- **Selected files**: `sglang_restomod/kernels/deep_gemm_sm70/__init__.py`.
  Installation point: `register()`, guarded by "no real `deep_gemm` resolves" and
  `device_sm == 70`; it marks the module `__restomod_substitute__`. Reachable in
  the engine only with patch 0009 **and** `SGLANG_SM70_DEEPGEMM_SUBSTITUTE=1`.
- **Hardware**: V100-SXM2-32GB, CUDA 12.8, torch 2.9.1+cu128. Evidence for the
  behaviours claimed here: `docs/runs/2026-11-17-prenorm-kernel-gate.md` and
  `docs/runs/2026-11-17-indexer-kernel-first-run.md`; the prepare-only half of
  patch 0008's reproduction is at
  `/nfs/ml_lab/projects/ml_lab/pding/scratch/dsv4-store-probe/repro_0008.sh`.
- **Status**: `local-only`.
- **Receipt note (2026-11-17)**: a PR touching `sglang_restomod/kernels/` needs a
  GPU receipt, but the policy job in `.github/workflows/restomod-gpu-receipt.yml`
  reports `requires=false` for it — the path alternation at line 106 lists
  `(quant|attention|moe|models|spec|linear_attn_sm70)` and omits `kernels/`.
  Until that list is fixed, a green `gpu-receipt` is **not** evidence for this
  file; treat the receipt as required.
