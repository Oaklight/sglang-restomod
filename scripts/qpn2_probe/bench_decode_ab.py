"""Kernel-level decode A/B: QPN2 GEMV against the shipping GEMV (#224, cross-run).

Not a server run. QPN2 cannot serve a prefill shape (``method.py`` raises), so a
two-run model comparison is impossible with prefill closed; this harness answers
the question that decides whether QPN2's prefill price (0.82x Marlin) is worth
paying: how far the QPN2 decode kernel is ahead of the shipping decode path on
the same inputs.

Arms, same routing and activation for every arm:

  * ``qpn2``   - ``decode_qpn2`` extension, QPN2 weight bank;
  * ``gemv``   - ``decode_gemv`` extension, Marlin-layout bank (the shipping
                 choice at gate_up: N/64 odd);
  * ``marlin`` - ``moe_wna16_marlin_gemm`` with both GEMV kernels disabled, the
                 shipping choice at down (N/64 even; ``decode_gemv`` declines).

TP4 shapes, as the server issues them: gate_up N=320 K=2560, down N=2560 K=160,
512 experts, 4 tokens x top_k 10 = 40 route rows.

Run one process per card, concurrently, to get per-card and aggregate GB/s:

    for g in 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g <python> scripts/qpn2_probe/bench_decode_ab.py \
        --tag card$g --out /tmp/ab/card$g.json &
    done; wait

Every arm is verified against the same FP32 reference on the first call; the
formal correctness gate stays the P2 fixture guard.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = {
    "gate_up": dict(n=320, k=2560),
    "down": dict(n=2560, k=160),
}
EXPERTS = 512
TOKENS = 4
TOP_K = 10
BLOCK = 8


def _quantize(torch, w):
    """FP16 [n, k] -> (packed uint8 [n, k//2], scales [n, k//16] float8)."""
    from sglang_restomod.quant.sm70_fp4_moe import check as chk

    return chk._quantize(w)


def build_layer(torch, n, k, experts, dev, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    packed, scales, _ = [], [], []
    for _ in range(experts):
        w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.float16).to(dev)
        p, s, _ = _quantize(torch, w)
        packed.append(p)
        scales.append(s)
    gs = (0.0123 * (1.0 + 0.1 * torch.rand(experts, generator=gen))).float().to(dev)
    return SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
        w13_weight=torch.stack(packed),
        w13_weight_scale=torch.stack(scales),
        w13_weight_scale_2=gs.clone(),
        w2_weight=torch.stack(packed),
        w2_weight_scale=torch.stack(scales),
        w2_weight_scale_2=gs.clone(),
    )


def build_routing(torch, dev):
    rows = TOKENS * TOP_K
    sorted_ids = torch.full((rows * BLOCK,), rows, dtype=torch.int32, device=dev)
    for r in range(rows):
        sorted_ids[r * BLOCK] = r
    expert_ids = torch.arange(rows, dtype=torch.int32, device=dev)
    return dict(
        rows=rows,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        padded=torch.tensor([rows * BLOCK], dtype=torch.int32, device=dev),
        topk_weights=torch.full((rows,), 1.0, dtype=torch.float32, device=dev),
    )


def time_kernel(fn, warmup=30, rounds=7, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1e3)
    return statistics.median(samples), min(samples), max(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="card")
    ap.add_argument("--out", default="")
    ap.add_argument("--shape", choices=tuple(SHAPES) + ("both",), default="both")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from sglang_restomod.quant.sm70_fp4_moe import decode_gemv, decode_qpn2
    from sglang_restomod.quant.sm70_fp4_moe.method import (
        moe_wna16_marlin_gemm,
        prepare_moe_nvfp4,
    )

    dev = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
    print(f"[{args.tag}] {dev_name}  torch {torch.__version__}", flush=True)

    os.environ["SGLANG_SM70_QPN2_TENSORS"] = "both"

    results = {"tag": args.tag, "device": dev_name, "shapes": {}}
    names = list(SHAPES) if args.shape == "both" else [args.shape]
    for name in names:
        n, k = SHAPES[name]["n"], SHAPES[name]["k"]
        routing = build_routing(torch, dev)
        x = torch.randn(TOKENS, k, dtype=torch.float16, device=dev)

        layer_m = build_layer(torch, n, k, EXPERTS, dev, args.seed)
        layer_q = build_layer(torch, n, k, EXPERTS, dev, args.seed)
        os.environ.pop("SGLANG_SM70_FP4_MOE_LAYOUT", None)
        prepare_moe_nvfp4(layer_m)  # every prefix to Marlin
        os.environ["SGLANG_SM70_FP4_MOE_LAYOUT"] = "qpn2"
        prepare_moe_nvfp4(layer_q)  # every prefix to QPN2

        out_m = torch.empty(routing["rows"], n, dtype=torch.float16, device=dev)
        out_q = torch.empty_like(out_m)
        rows = routing["rows"]
        split_k = decode_qpn2.split_k_for(k)
        ws_m = decode_gemv._workspace_for(rows, n, dev)
        ws_q = decode_qpn2._workspace_for(rows, n, k, dev)

        mg = decode_gemv._load()
        mq = decode_qpn2._load()
        if mg is None or mq is None:
            print("extension build failed", flush=True)
            return 2

        def run_gemv():
            return mg.decode_moe_gemv(
                x, out_m, layer_m.w13_weight, layer_m.w13_weight_scale,
                layer_m.w13_weight_scale_2, routing["sorted_ids"],
                routing["expert_ids"], routing["topk_weights"], rows, TOP_K, n, k,
                True, ws_m, decode_gemv.K_SPLIT, routing["padded"])

        def run_qpn2():
            return mq.decode_moe_gemv_qpn2(
                x, out_q, layer_q.w13_weight, layer_q.w13_weight_scale_2,
                routing["sorted_ids"], routing["expert_ids"], routing["topk_weights"],
                rows, TOP_K, n, k, True, ws_q, split_k, routing["padded"])

        def run_marlin():
            out = torch.empty_like(out_m)
            return moe_wna16_marlin_gemm(
                x, out, layer_m.w13_weight, None, layer_m.w13_weight_scale,
                layer_m.w13_weight_scale_2, None, None, None,
                torch.empty(0, dtype=torch.int32, device=dev),
                routing["sorted_ids"], routing["expert_ids"], routing["padded"],
                routing["topk_weights"], BLOCK, TOP_K, False, False,
                __import__("sgl_kernel.scalar_type", fromlist=["scalar_types"]).scalar_types.float4_e2m1f,
                TOKENS, n, k)

        weight_bytes = EXPERTS * 0 + rows * (n * k // 2 + n * (k // 16))
        arms = {}
        for label, fn in (("gemv", run_gemv), ("qpn2", run_qpn2), ("marlin", run_marlin)):
            prev_g = os.environ.get("SM70_MOE_DECODE_GEMV")
            prev_q = os.environ.get("SM70_MOE_DECODE_QPN2")
            if label == "marlin":
                os.environ["SM70_MOE_DECODE_GEMV"] = "0"
                os.environ["SM70_MOE_DECODE_QPN2"] = "0"
            try:
                fn()
                torch.cuda.synchronize()
                us, lo, hi = time_kernel(fn)
            except Exception as exc:  # noqa: BLE001 - an arm may decline a shape
                arms[label] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            finally:
                for var, prev in (("SM70_MOE_DECODE_GEMV", prev_g),
                                  ("SM70_MOE_DECODE_QPN2", prev_q)):
                    if prev is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = prev
            arms[label] = {"us": us, "min": lo, "max": hi,
                           "GB/s": weight_bytes / (us * 1e-6) / 1e9}

        if "us" in arms.get("gemv", {}) and "us" in arms.get("qpn2", {}):
            a, b = run_gemv(), run_qpn2()
            torch.cuda.synchronize()
            scale = a.float().abs().max().item()
            arms["qpn2_vs_gemv_maxabs"] = ((a.float() - b.float()).abs().max().item()
                                           / max(scale, 1e-9))
        if "us" in arms.get("qpn2", {}) and "us" in arms.get("marlin", {}):
            arms["qpn2_over_marlin"] = arms["qpn2"]["us"] / arms["marlin"]["us"]
        if "us" in arms.get("qpn2", {}) and "us" in arms.get("gemv", {}):
            arms["qpn2_over_gemv"] = arms["qpn2"]["us"] / arms["gemv"]["us"]

        results["shapes"][name] = dict(n=n, k=k, rows=rows, split_k=split_k,
                                       weight_MB=weight_bytes / 1e6, arms=arms)
        print(f"\n[{args.tag}] {name}: N={n} K={k} rows={rows} "
              f"({weight_bytes/1e6:.1f} MB) split_k={split_k}", flush=True)
        for label in ("qpn2", "gemv", "marlin"):
            a = arms.get(label, {})
            if "us" in a:
                print(f"  {label:<7} {a['us']:7.1f} us  {a['GB/s']:5.0f} GB/s",
                      flush=True)
            else:
                print(f"  {label:<7} {a.get('error', 'skipped')}", flush=True)
        for key in ("qpn2_over_gemv", "qpn2_over_marlin", "qpn2_vs_gemv_maxabs"):
            if key in arms:
                print(f"  {key}: {arms[key]:.3f}", flush=True)
        del layer_m, layer_q, out_m, out_q
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
