# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Every entry cites the pull request that carries it, so a reader can check it in
one click.

Nothing has been released yet: `__version__` is `0.0.0a0.dev0` and the only tags
in the repository are signed evidence bundles. There is therefore one
`Unreleased` section and no release dates.

Scope: the file begins at the 2026-10-23 rebuild as a lightweight downstream
layer. It records what a downstream user would notice, not every commit.
Measurement records, with their hardware, arm and protocol, are in
[`docs/runs/`](docs/runs/); validated configurations are in
[`COMPATIBILITY.md`](COMPATIBILITY.md).

Entries marked **†** merged with the `gpu-receipt` check red. They carry the
tests their pull request names, but no signed evidence bundle, so they are not
validated on hardware.

## [Unreleased]

### Added

- `attest` records a component whose module fails to import as `inactive` with
  the import error; the loader can no longer report "loaded" while the engine
  serves its own backend (#297).
- `restomod doctor` fails when the checkout's recorded commit does not match
  the declared `[engine].commit`; a stale site tree is an error, not a
  configuration choice (#297).
- `BINARIES.toml`, `restomod doctor`, and `scripts/fetch_binaries.py` /
  `scripts/build_binaries.py`: pin the out-of-repo artefacts (prebuilt
  libraries, flashinfer fork, container image) by sha256 and provenance, and
  fail loudly when a selected component's artefact is missing or mismatched
  (#295)†.
- The hand-tuned V100 MoE tiles (E=512, N=160) and the QPN8 decode source now
  travel with the wheel; `SGLANG_SM70_QPN8_SRC` is an override, not a
  requirement (#295)†.
- Opt-in component loading through `SGLANG_RESTOMOD_COMPONENTS`; with the
  variable unset the plugin loads and changes nothing (#56).
- `fp8_moe`: block-FP8 expert GEMM bound from the prebuilt TurboMind library
  (#56, #57).
- `attention_v100`: V100 full-attention backend with the vendored TileLang FA
  kernels and the `tilelang_fa_v100` backend name (#88), plus the CUDA-graph
  metadata hooks (#58).
- `fp8_dense`, `fp4_moe`, `ple_sm70` and `spec_sm70`, which together serve
  Flash-Next at the pin (#96).
- `fp8_lm_head`, which quantizes the LM head to block FP8 after load and
  requires `fp8_dense` (#157).
- `fp4_dense`: dense NVFP4 through the `marlin_v100` extension (#178).
- `sm70_marlin_v100`: dense INT4 WNA16 through the `marlin_v100` extension
  (#159).
- `fp8_dense_quant`, which quantizes a BF16 checkpoint's dense linears to block
  FP8 after load (#191).
- `moe_align_sm70`, which routes tiny-batch MoE alignment to the CUDA pair
  instead of the single-CTA Triton kernel (#199).
- `gdn_tilelang_sm70`, TileLang chunked prefill for GDN linear attention
  (#106).
- `lsa_sm70`, Latent Sparse Attention for SM70, later reduced to a registration
  shim (#181, #215).
- A decode-shaped NVFP4 MoE GEMV for the shapes it can read exactly (#202).
- A patch queue with three upstream edits: a HiCache/OpenSSL fallback (#86), a
  `cuda_ipc` JIT fix without which custom allreduce is silently disabled (#98),
  and a QPN2 shape gate in `fused_marlin_moe` (#246).

### Changed

- The engine baseline is v0.5.23; `pyproject.toml` declares `sglang==0.5.23` and
  `sglang-kernel` (#243, #267, #268).
- The plugin records which components registered, and one failing component no
  longer takes the others down with it (#278)†.
- The prebuilt TurboMind library is discovered without a fork-only module or a
  required environment variable; an environment variable is accepted only when
  it names an existing file (#286)†.
- The GDN TileLang pair is bound at the FLA wrappers. `SM70_GDN_TILELANG=0`
  keeps the pinned Triton path (#275)†.
- FP8 shapes the packed E4M3 kernel cannot represent are refused instead of
  computed (#264).
- `fp8_dense_quant` admits `RowParallelLinear` by class rather than by one
  attribute, so Flash-Next's row-parallel modules are converted (#230).
- NVFP4 MoE split-K is chosen per launch and follows the CTA count
  (#183, #197).
- MTP draft decode runs on the native split-KV kernel instead of the Triton
  container (#223).
- Wheels include the `fp4_moe` CUDA source, the marlin patch manifest and the
  GDN provenance (#281).

### Fixed

- `restomod doctor` says what a failed check found instead of reusing the label
  for the passing state, so a commit mismatch prints "does not match" rather
  than "matches" (#303).
- `scripts/redact_evidence.py` keeps the bundle's redaction record across
  passes: a second pass that removes nothing no longer rewrites every counter
  to zero, and the rules no longer re-match their own output (#303).
- A suite pack reports the archive's own sha256 beside the manifest digest
  `oob pack` prints, because the manifest digest cannot say which directory a
  person means (#303).
- The GDN tail chunk is padded and trimmed instead of refused, so prompts whose
  token count is not a multiple of 64 stay on TileLang (#287)†.
- A decode-GEMV skip is logged once per shape instead of once per layer (#269)†.
- The FP8 LM head is built through its constructor (#255).
- The speculative JIT builds under CUDA 12.x with a CCCL shim for the pinned
  `flashinfer/sampling.cuh` (#179).
- Attestation writes use their own temporary file instead of sharing one
  (#290)†.
- A missing kernel library is reported as a path error instead of raising from
  `Path(None)` (#264).

### Removed

- The vendored LSA implementation. `lsa_sm70` is now a shim that registers the
  standalone `Oaklight/lsa-sm70` package (#215).

### Breaking changes

- `lsa_sm70` requires the standalone `lsa-sm70` package at the commit pinned in
  `UPSTREAM_PIN.toml`. Installing this layer alone no longer provides an LSA
  implementation (#215).
- The engine baseline is v0.5.23. An environment that pins v0.5.22 is no longer
  the baseline against which the components are built (#243, #267).
- A source build applies the three-patch queue. A v0.5.23 tree without the
  `cuda_ipc` patch compiles, but custom allreduce is silently disabled (#98).

[Unreleased]: https://github.com/Oaklight/sglang-restomod/commits/main
