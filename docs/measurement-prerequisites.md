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

2. **Point `SM70_MARLIN_DIR` at the dense build.** It must hold
   `_sm70_marlin_v100_dense.abi3.so` (~276 MB). The 1.2 MB `_C.abi3.so`
   exports `marlin_gemm` with no CUDA implementation; loading it fails with a
   `NotImplementedError` that reads like a broken test. Check the file size
   before the run.

3. **Export `SM70_TURBOMIND_SO`** (and `SGLANG_SM70_TURBOMIND_SO`). Without it,
   three fp8_moe tests die at import.

4. **Put `SGLANG_V100_DECODE_CUDA_BUILD_DIR` on a filesystem with space.** The
   default lands on a near-full root. Since #244 the preflight refuses below
   1 GiB. That refusal is correct: free space, do not chase it as a test
   failure.

## Measurement

5. **Leave room in `--contexts` for the output.** A prompt of exactly the
   context length, with generated tokens on top, returns HTTP 400. Use 3,900
   against an 8,192 context.

6. **Take tree hashes from the script.** They exclude `__pycache__`, and the
   composite digest is locale-pinned inside the script. Do not re-derive them
   with your own `find`/`sort`: a hash that depends on the caller's environment
   is not a gate.

7. **Do not compare load time or page-cache state across sessions.** The same
   checkpoint took 132.7 s and 389.3 s to load; that is I/O, not the code under
   test. An arm whose load straddles external I/O is not the same arm. Run no
   bulk filesystem I/O while a measurement session is active.
