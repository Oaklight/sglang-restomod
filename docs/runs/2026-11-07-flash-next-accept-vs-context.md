# 2026-11-07 — Flash-Next accept length does not fall with context

Status: **accept length and verify step time are flat from 3.8k to 30.8k.**
Acceptance rises slightly (3.879 → 4.000), step time moves by 0.4 ms across an
8x context change (31.96 → 32.40 ms median). Neither the draft nor the
verification degrades in this range.

## What was run

The receipted Flash-Next NVFP4 shape: 4x V100-SXM2-32GB on one island
(lambda4 GPUs 4-7), TP4, `--ep-size 1`, `--ple-offload-embedding`, MTP
`--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4` with rejection sampling, greedy, page size
64, MTP on. Components: `attention_v100,fp4_moe,moe_align_sm70,ple_sm70,spec_sm70`.

| | |
|---|---|
| Tree | main `4add639ac9` (tree `55c91052a9`), extracted with `git archive`, no `__pycache__` |
| Engine | pin `v0.5.23` @ `297e54fd`, `pin-site-v0.5.23` |
| Host | lambda4 GPUs 4-7, driver 550.163.01, CUDA 12.8 image |
| Kernels | issue178 marlin build (`work/issue178-5c0cbbaf/marlin-ops`), `SM70_TURBOMIND_SO` |
| Build dir | `SGLANG_V100_DECODE_CUDA_BUILD_DIR` on the ml_lab mount, `TMPDIR=/dev/shm` |

Entry inventory: four cards at 1 MiB, no compute apps, no server process.

One `/generate` per rep, 128 output tokens, `temperature 0`, `ignore_eos`,
streamed; `spec_accept_length` and `spec_verify_ct` are read from the same
response as the timings. Step time is `decode_s / spec_verify_ct`, so it does
not depend on how acceptance is estimated. Prompt is the bench seed text
repeated to the requested length, identical across cells.

## Results

| Cell | Context | Prompt tokens | Accept | Verify passes | Step ms (reps) | Median step | Decode tok/s |
|---|---|---:|---:|---:|---|---:|---:|
| 4k | 4,096, mf 0.82 | 3,779 | 3.879 | 33 | 31.37 / 31.96 / 32.08 | 31.96 | 123.6 / 121.4 / 120.9 |
| 8k | 8,192, mf 0.80 | 7,654 | 4.000 / 3.879 / 3.879 | 32 / 33 / 33 | 31.48 / 32.29 / 32.20 | 32.20 | 127.1 / 120.1 / 120.5 |
| 32k | 32,768, mf 0.82 | 30,807 | 4.000 | 32 | 31.68 / 32.40 / 32.48 | 32.40 | 126.3 / 123.5 / 123.2 |

Pool at the 32k launch: `max_total_num_tokens=201,920`, 6.19 GB spare, so the
32k cell fits on four cards; it is not a memory-limited measurement.

## Reading

- **Draft:** acceptance does not degrade with context. It is 3.879 at 3.8k and
  4.000 at 30.8k. The 8k cell's mixed 4.000/3.879 is the first-token path, not
  a context trend.
- **Verification:** one verify pass per step, and the step is 31.96 to 32.40 ms
  across the range. The 0.4 ms spread is the same size as the rep-to-rep spread
  inside a cell, so there is no measurable context term here.
- The earlier noise-reference observation that a 4k-to-7.9k decode gap was
  acceptance rather than kernel time is consistent with this: step time is
  context-flat, so a decode-rate difference at fixed step time can only come
  from accept length. In this run accept length did not fall, so decode rate
  did not fall either.

## Caveats

- The prompt is repeated seed text. Acceptance is content-dependent, and a
  3.879 vs 4.000 difference between cells is a prompt artifact; the absence of
  a downward trend is the finding, not the exact value.
- Memory fraction differs between cells (0.82 / 0.80 / 0.82) because each cell
  is its own launch. That changes pool size, not kernel time.
- **Overlap:** the lambda6 GPU suite (`df`) ran on the same checkpoint through
  the same NFS filer for the whole window (11:58 24.9 GB, 12:10 30-31 GB across
  four cards). My cells ran 11:47-12:10. Decode is 3-5% below the 127 tok/s
  greedy reference; that gap is consistent with the shared filer, and the
  within-run comparison is unaffected because all three cells saw the same
  background. Re-measure one cell on an idle pair of hosts before quoting an
  absolute number.
