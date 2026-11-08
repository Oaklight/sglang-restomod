# Measurement prerequisites

Read this before running `scripts/gpu_suite.sh` or signing a receipt. A green
check on a host that misses these describes a different machine or a different
measurement. Every item below came from a run where the failure did not look
like one.

## Host

1. **Use V100 / sm_70 only.** Confirm the device before the suite:

   ```sh
   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
   # Tesla V100-SXM2-32GB, 7.0
   ```

   Do not use lambda5: it is a T4 (sm_75), and the SM70 tests self-skip there,
   so the receipt is green and about the wrong machine. Check the device name,
   not free memory.

2. **Point `SM70_MARLIN_DIR` at the dense build.** It must hold the dense
   extension file (~276 MB); item 6 says how to identify it, and names the
   failure a wrong file produces.

3. **Export `SM70_TURBOMIND_SO`** (and `SGLANG_SM70_TURBOMIND_SO`). Without it,
   three fp8_moe tests die at import.

4. **Put `SGLANG_V100_DECODE_CUDA_BUILD_DIR` on a filesystem with space.** The
   default lands on a near-full root. Since #244 the preflight refuses below
   1 GiB. That refusal is correct: free space, do not chase it as a test
   failure.

5. **Run against the sglang tree for the pin.** Point `PYTHONPATH` at the tree
   that matches `UPSTREAM_PIN.toml`, not the previous pin's; for the current
   pin the canonical tree is `pin-site-v0.5.23` under the shared `ml_lab`
   volume. A mismatched tree fails at collection with `ImportError` on a
   symbol the newer pin has (for example `sparse_gqa_paged_prefill_triton`):
   a missing symbol at collection, not a test failure, and it reads as the
   change under test breaking things. Confirm the bundle's `versions.txt`
   records the pin's sglang version.

6. **Identify the dense extension by the file, not by its name.** A site can
   carry the right name and the wrong file. Check the size on disk in the
   directory `SM70_MARLIN_DIR` points at, whatever the files are called there
   (a receipt passed with `_C.so` at 276,745,992 B and `_moe_C.so` at
   258,397,688 B). The failure this prevents: a site whose names look right but
   whose file is the 1.2 MB stub, which exports `marlin_gemm` with no CUDA
   implementation and fails with a `NotImplementedError` that reads like a
   broken kernel.

7. **Write the evidence directory outside the checkout.** `gpu_suite.sh`
   measures the worktree before it creates `--out`, so an in-repo default marks
   the run `dirty_files=1` even though `evidence/` is generated, and the next
   receipt then needs `--allow-dirty`. Point `--out` at a directory on the data
   volume (for example `.../builds/evidence-<head>`), and remove any stale
   `evidence/` or `evidence.tgz` left in the checkout by an earlier attempt
   before starting. The failure this prevents: a green run you cannot sign, or
   a second run paid to clear a leftover file rather than a source change.

## Measurement

8. **Leave room in `--contexts` for the output.** A prompt of exactly the
   context length, with generated tokens on top, returns HTTP 400. Use 3,900
   against an 8,192 context.

9. **Take tree hashes from `scripts/tree_hash.sh`.** It covers
   `UPSTREAM_PIN.toml`, `patches/` and `sglang_restomod/`, excludes
   `__pycache__`, and pins the locale inside the script. Record the digest with
   the commit it belongs to, and check a later tree against it with
   `scripts/tree_hash.sh <tree> --expect <digest>`. Do not re-derive the value
   with your own `find`/`sort`: a hash that depends on the caller's environment
   is not a gate.

10. **Do not compare load time or page-cache state across sessions.** The same
   checkpoint took 132.7 s and 389.3 s to load; that is I/O, not the code under
   test. An arm whose load straddles external I/O is not the same arm. Run no
   bulk filesystem I/O while a measurement session is active.
