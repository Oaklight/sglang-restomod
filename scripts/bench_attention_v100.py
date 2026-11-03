#!/usr/bin/env python3
"""Standalone microbench of the attention_v100 paged extend kernel.

Times `paged_forward` for a single request at several prefix lengths, with a
number of extend rows that matches a chunked prefill. Reports ms per call,
the KV bytes the call must read, and the effective read rate, so the number
can be compared against the card's copy ceiling instead of against nothing.

    python scripts/bench_attention_v100.py
    python scripts/bench_attention_v100.py --prefixes 4096,16384,65536

Needs the SM70 environment (torch with CUDA and tilelang); run it inside the
CUDA 12.8 image on a V100.
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from sglang_restomod.attention.v100.tilelang import paged_forward


def build(prefix, extend, heads, heads_kv, dim, block, device):
    total = prefix + extend
    blocks = (total + block - 1) // block
    k_cache = torch.randn(blocks, block, heads_kv, dim, dtype=torch.float16, device=device)
    v_cache = torch.randn(blocks, block, heads_kv, dim, dtype=torch.float16, device=device)
    q = torch.randn(extend, heads, dim, dtype=torch.float16, device=device)
    out = torch.empty_like(q)
    args = (
        k_cache,
        v_cache,
        torch.arange(blocks, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor([total], dtype=torch.int32, device=device),
        torch.tensor([0, extend], dtype=torch.int32, device=device),
        torch.tensor([prefix], dtype=torch.int32, device=device),
    )
    kwargs = dict(
        out=out,
        block_size=block,
        softmax_scale=1.0 / math.sqrt(dim),
        causal=True,
        num_kv_heads=heads_kv,
    )
    return q, args, kwargs, total


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(3):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best.append(start.elapsed_time(end) / iters)
    return min(best), statistics.median(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="1024,4096,16384,65536")
    ap.add_argument("--extend", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--heads-kv", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    device = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"gpu: {name} | extend {args.extend} rows, heads {args.heads}/{args.heads_kv}, dim {args.dim}")
    print()
    print(f"{'prefix':>8} {'best ms':>9} {'med ms':>8} {'KV MiB':>9} {'GB/s':>7}")

    for prefix in (int(x) for x in args.prefixes.split(",")):
        q, args_, kwargs, total = build(
            prefix, args.extend, args.heads, args.heads_kv, args.dim, args.block, device
        )
        call = lambda: paged_forward(q, *args_, **kwargs)  # noqa: E731
        call()
        torch.cuda.synchronize()
        best, med = time_call(call, args.warmup, args.iters)
        kv_bytes = total * 2 * args.heads_kv * args.dim * 2  # K and V, fp16
        gbps = kv_bytes / (best * 1e-3) / 1e9
        print(f"{prefix:8d} {best:9.3f} {med:8.3f} {kv_bytes / 2**20:9.1f} {gbps:7.1f}")


if __name__ == "__main__":
    main()
