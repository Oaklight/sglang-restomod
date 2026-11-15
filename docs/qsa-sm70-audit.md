# QSA on SM70: audit and coverage

Audit of 2026-11-06, re-verified 2026-11-15 at `863e1a15`. The original audit
was issue #232; this file replaces it so the findings survive the issue. Where
the audit's conclusion has been overtaken since, this file says so instead of
carrying the old text forward.

Scope: what the QSA path in this component is, where it diverges from the pin,
what has been exercised on hardware, and what remains unowned. No GPU run was
made for this document.

## 1. What exists

Sourced: read from this tree at `863e1a15`.

| piece | path | lines |
|---|---|---:|
| QSA backend | `sglang_restomod/attention/v100/qwen_sparse_attn_backend.py` | 2,568 |
| registration and class swap | `sglang_restomod/attention/v100/backend_registry.py` | 162 |
| draft factory | `sglang_restomod/attention/v100/draft_backend.py` | 150 |
| CUDA bindings | `sglang_restomod/attention/v100/tilelang/_decode_cuda.py` | 409 |
| process-wide QSA patches | `sglang_restomod/overrides.py` | 667 |
| numeric kernel tests | `tests/test_qsa_kernels_gpu.py` | 352 |

The engine side (pin `sgl-project/sglang` v0.5.23, `297e54fd`) carries
`srt/layers/attention/qsa/` and its own `qwen_sparse_attn_backend.py`. No `qsa/`
package is vendored here, and that remains the right call: there is no
interface break to fix, the pin supplies it, and vendoring it would add about
3k lines to maintain.

## 2. What changed since the audit

Sourced: `git log`.

- **#243 (`e1b53e8eb3`, v0.5.23 bump) rebased the vendored backend onto the
  pin's own file.** The audit asked for this "on the next pin bump"; it
  happened on that bump. The fork-lineage copy and its eleven drift hunks are
  gone, and the pin-bump also adopted upstream #45752 (prefill indexer gather)
  and #45791 (long cached prefixes).
- **The same change closed the `PROVENANCE.md` staleness the audit found.** The
  provenance now records that the combine comes from `tilelang/_decode_cuda.py`
  and that `sglang.kernels.ops.attention.sm70_qsa_combine` and
  `SGLANG_SM70_QSA_COMBINE` do not exist at the pin.
- **#235 (`c2ba3297e7`) added numeric kernel tests**, which removes the audit's
  largest finding ("no numeric or GPU kernel test for any QSA kernel path").

The audit's measurements of the fork's line counts and hunk offsets are
historical; do not use them to compare against the current file.

## 3. Coverage now

Sourced: `tests/test_qsa_kernels_gpu.py`, `COMPATIBILITY.md`, `recipes/`.

`tests/test_qsa_kernels_gpu.py` covers the kernel boundaries numerically,
against a dequantized reference, and runs under `pytest -m gpu tests/`:

| test | what it exercises |
|---|---|
| `test_sparse_decode_e5m2_matches_reference[6]`, `[3]` | SM70 sparse decode, e5m2 KV, the two supported head groups |
| `test_sparse_decode_fp16_matches_reference` | the fp16 KV path |
| `test_sparse_prefill_e5m2_matches_reference` | the sparse chunk-prefill e5m2 path |
| `test_dense_paged_decode_e5m2_cuda_matches_reference[occ2]`, `[serial]` | dense paged decode, CUDA variant |
| `test_dense_paged_decode_e5m2_tilelang_matches_reference` | the same through TileLang |
| `test_indexer_decode_fp16_matches_reference` | the indexer decode binding |

So the two specializations the audit called "selectable but never exercised"
now have numeric tests, and the "dead binding" has a test caller.

What is still missing is a different thing: **no engine-level run is recorded
for either specialization.** A numeric kernel test says the kernel computes
what it claims on the fixture it is given; it says nothing about the model path
that selects it. The recorded engine runs remain the ones the audit listed:

| dimension | recorded | not recorded |
|---|---|---|
| decode | TP4 NVFP4 page 64 MTP (receipts), TP8 FP8 4.1k/8.0k (lab, #93) | page size 16 with e5m2 KV |
| prefill | TP4 NVFP4 dense prefill (lab), TP8 FP8 4.1k/8.0k (lab) | the sparse chunk-prefill route on e5m2 KV |
| KV dtype | fp16 (the pin default, all receipts) | e5m2 on a Flash-Next engine run; e4m3 is refused on Volta |
| long context | 4,096 in `evidence-cbc56116-splitk` | any QSA receipt above 4k |

## 4. What still stands

Sourced (code read at `863e1a15`), with the consequence inferred.

1. **The QSA class swap is unconditional.** `backend_registry.py:75
   install_qsa_swap()` rebinds the pin's module attribute on import success and
   an idempotence flag alone: no hardware check, no model check, no env
   condition. An `is_hip()` or capability guard would make a wrong-machine
   selection visible instead of letting the component replace the engine class
   everywhere. Consequence: inferred; the audit did not measure it.
2. **`sm70_cuda_qsa_indexer_decode` has a test caller and no production
   caller.** `_decode_cuda.py:385` defines it; `tests/test_qsa_kernels_gpu.py`
   calls it; nothing in the model path does. Either wire it or remove it — an
   untested binding is a liability, and a tested-but-unused one is dead weight.
3. **The long-context row still wants a receipt.** `COMPATIBILITY.md:28` records
   84.69–99.43 tok/s at 31.7K for Flash-Next NVFP4 as a lab record with no
   receipt, and `recipes/flash-next-nvfp4-v100-tp4-mtp.md:258` says contexts
   above 4k are covered by a lab record only. Reconcile the two when #95's
   receipt exists.

## 5. Triggers

Recorded outside this file on purpose; a deferral that lives only in a closed
document is never picked up.

- **Pin bump**: re-check the vendored QSA backend against the pin's file and
  re-apply the SM70 routes. Listed on the pin-bump steps in `CONTRIBUTING.md`.
- **#95 receipt**: reconcile the 31.7K lab row against the recipe's "above 4k
  is lab only" note, and record whichever is true. Noted on #95.

## 6. Verdict

Sourced claims above support it; the judgment is inferred.

Nothing must be ported, and nothing must be vendored. The backend follows the
pin as of #243, the kernel boundaries are numerically tested as of #235, and
what remains is the engine-level evidence the receipts do not yet carry. The
two small cleanups in section 4 should ride with the next change under
`sglang_restomod/` so they do not justify a receipt of their own.
