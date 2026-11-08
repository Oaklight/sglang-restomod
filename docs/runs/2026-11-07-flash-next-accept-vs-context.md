# 2026-11-08 — Flash-Next accept length does not fall with context

Status: **accept length and verify step time are flat from 3.8k to 30.8k.**
Acceptance is 3.879 at 4k and 4.000 at 32k; step time is 27.08, 27.30 and
27.38 ms median. Neither the draft nor the verification degrades in this range.

## What was run

The receipted Flash-Next NVFP4 shape: 4x V100-SXM2-32GB on one island
(lambda6 GPUs 0-3), TP4, `--ep-size 1`, `--ple-offload-embedding`, MTP
`--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4` with rejection sampling, greedy, page size
64. Components: `attention_v100,fp4_moe,moe_align_sm70,ple_sm70,spec_sm70`.

| | |
|---|---|
| Tree | main `39d4bc7b`, engine digest `44bfdfa4…` (`scripts/tree_hash.sh`, pin 297e54fd) |
| Engine | v0.5.23 site with all three patches, at `work/pin-site-v0.5.23-patched`; `ipc.cuh` sha256 `eb19c88b…` |
| Host | lambda6 GPUs 0-3, driver 550.163.01, CUDA 12.8 image; entry inventory all four cards at kmarlor baseline, no server process |
| Kernels | issue178 marlin build (`_C.so`, 276,745,992 B), `SM70_TURBOMIND_SO` and `SGLANG_SM70_TURBOMIND_SO` |

Custom allreduce: zero `Setup Custom allreduce failed` lines in all three
cells, and the JIT cache gained `custom_all_reduce_fp16_t_4_false` at this
launch. The cells are therefore comparable to each other; without that, a
~3 ms NCCL-ring fallback could sit in one cell and not another.

Step is `decode_s / spec_verify_ct`: one verify pass per step, `decode_s` is
the wall time from the first non-empty streamed chunk to the last in the same
`/generate` response that carries `spec_accept_length` and `spec_verify_ct`, so
prefill is excluded.

## Results

| Cell | Context / mf | Prompt tokens | Accept | Verify passes | Step ms (reps) | Median | Decode tok/s |
|---|---|---:|---:|---:|---|---:|---:|
| A | 4,096 / 0.82 | 3,779 | 3.879 | 33 | 26.59 / 27.08 / 27.21 | 27.08 | 145.9 / 143.2 / 142.6 |
| B | 8,192 / 0.80 | 7,654 | 3.879 / 3.765 / 3.879 | 33 / 34 / 33 | 26.79 / 27.30 / 27.39 | 27.30 | 144.8 / 137.9 / 141.6 |
| C | 32,768 / 0.82 | 30,807 | 4.000 | 32 | 27.05 / 27.38 / 27.63 | 27.38 | 147.9 / 146.1 / 144.8 |

The 32k cell was served, not declined: the launch reports
`max_total_num_tokens=116,928` with the pool sized at 0.82, and the 30,807-token
request decoded normally. That number is what makes the flat result cover
32k rather than stopping at 8k.

The memory fraction differs between cells because each cell is its own launch;
it changes the pool, not the kernel.

## Reading

- **Draft:** acceptance does not fall with context. 3.879 at 3.8k, 3.879-4.000
  at 7.7k, 4.000 at 30.8k. One B rep came in at 3.765, i.e. 34 verify passes
  instead of 33 — one step accepted one fewer token. The other two B reps
  match A exactly, so the cell value is unchanged; the single rep is named
  rather than smoothed.
- **Verification:** step rises by 0.22 ms (B) and 0.30 ms (C) over A. Both are
  inside the 0.71 ms threshold set below, and inside the 0.62 ms spread
  within A itself.
- The earlier observation that a 4k/7.9k decode gap was acceptance rather than
  kernel time holds: with step time context-flat, only acceptance can move
  decode rate, and acceptance did not fall.

**Falsifier, stated before the run:** the claim is falsified if any cell's
median step is more than 0.7 ms above A's in a monotone direction with context,
or if accept in B or C falls more than 0.1 below A's. The 0.7 ms is the larger
of two measured spreads: 0.71 ms in A of the 2026-11-07 run and 0.62 ms in A
of this run. Neither condition is met.

## Earlier reference (2026-11-07, not comparable)

The same three cells were run a day earlier on an overlapped island with the
unpatched v0.5.22 site: step 31.96 / 32.20 / 32.40 ms, accept 3.879-4.000.
Today is 27.1 ms and that run was 32.0 ms; the difference is **not** a context
effect. It is a different site (patched v0.5.23 against unpatched v0.5.22, with
a possible NCCL-ring fallback in the older arm) and an idle island against an
overlapped one, so the two are not comparable. The 2026-11-07 table is kept
only as a labelled historical record.

## Where the per-cell server configuration comes from

Cell A's `/get_server_info` was captured live (`server_info_A.json`). B and C
were stopped before that fetch, so their entry is the same `server_args` dict
extracted from the launch log (`server_args_B.json`, `server_args_C.json`),
without the endpoint's resolved values. The cells differ only in
`--context-length` and `--mem-fraction-static`; the launch command is
`launch.sh <ctx> <mf> <port>`, sha256 `7d1842d7…`.

## Caveats

- The prompt is repeated seed text. Acceptance is content-dependent; the
  absence of a downward trend is the finding, not the exact value.
- Absolute decode is 143-146 tok/s on this host at 4k. 2d measured a +2.0%
  host component for the same configuration on a different island, so an
  absolute number belongs with its host.
