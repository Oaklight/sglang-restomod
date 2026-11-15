#!/usr/bin/env python3
"""Collect and aggregate a per-step decode kernel budget for a served model.

Two modes:

    python scripts/trace_decode_budget.py collect \
        --host 127.0.0.1:11436 --model qwen38-flashnext-nvfp4 \
        --model-dir /path/to/checkpoint --out-dir /tmp/prof-decode

    python scripts/trace_decode_budget.py analyze /tmp/prof-decode/<trace>.json.gz

``collect`` measures the end-to-end decode rate on one unprofiled request, then
captures a decode-only window with the engine profiler and analyzes it.
``analyze`` classifies the kernels of an existing trace.

The trace is a PyTorch Kineto chrome trace. Kernel time is device time per
launch, so it is the right quantity for a "where does the step go" budget, and
the end-to-end rate is recorded beside it as the anchor that says the traced
window is the served configuration.

Step count comes from a per-MoE-layer anchor kernel rather than from the
profiler's ``num_steps``: the window can cover one more forward than requested,
and a wrong divisor shifts every row. The default is the Flash-Next shape
(48 target layers + 3 MTP draft steps = 51 calls per forward). Pass ``--steps``
to override it, or ``--per-step-anchors`` when another model has a different
layer count.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import threading
import time
import urllib.request
from collections import defaultdict

# --- kernel categories -----------------------------------------------------
# Order matters: a name is tested against these groups in order.
_COLLECTIVE = (
    "all_reduce_1shot_push_kernel",
    "ncclDevKernel_AllReduce",
    "ncclDevKernel_AllGather",
    "ncclDevKernel_ReduceScatter",
    "ncclDevKernel_Broadcast",
)
_ATTENTION = (
    "qsa_decode_partial_kernel",
    "qsa_index_q_prep_kernel",
    "qsa_index_k_compress_kernel",
    "_expand_qsa_block_indices_kernel",
    "_fused_qk_rmsnorm_rope_gate_kernel",
    "store_kvcache_kernel",
    "fused_sigmoid_gating_delta_rule_update_kernel",
    "_causal_conv1d_update_kernel",
    "_fused_sigmoid_mul_kernel",
    "fused_qkvzba_split_reshape_cat_contiguous_kernel",
    "_fused_mamba_state_scatter_with_mask_kernel",
    # TileLang emits generic names; on this model these are the decode and
    # linear-attention kernels. The detail row keeps them visible.
    "main_kernel",
    "kernel_kernel",
)
_NORM = (
    "grouped_gemma_rmsnorm_kernel",
    "RMSNormKernel",
    "_layer_norm_fwd_1pass_kernel",
    "rms_norm",
    "layer_norm",
)
_MOE_SUPPORT = (
    "moe_align",
    "count_and_sort_expert_tokens",
    "moe_sum_reduce",
    "act_and_mul",
    "_fused_gate_sigmoid_mul_add",
    "moe_permute",
)
_MOE_ROUTER = (
    "_router_triton_kernel",
    "fast_topk",
)
_DENSE_GEMM = (
    "turbomind::gemm::gemm_kernel",
    "cutlass::Kernel2<cutlass_70_wmma",
    "gemmSN_TN_kernel",
    "gemv2T_kernel_val",
    "cublasLt::splitKreduce_kernel",
    "cutlass::gemm::device::Gemm",
)
# Marlin tile spec fields that separate the two expert GEMMs on this profile.
_GATE_UP_TILE = "Sm70MoeNvfp4GemmSpec, 64, 64,"
_DOWN_TILE = "Sm70MoeNvfp4GemmSpec, 32, 256,"

# One call per MoE layer per forward: 48 target layers + 3 draft steps.
ANCHOR_KERNELS = ("_moe_align_small_numel_kernel", "moe_align_block_size_kernel")
DEFAULT_PER_STEP_ANCHORS = 51


def _has(name: str, needles: tuple[str, ...]) -> bool:
    return any(n in name for n in needles)


def classify(name: str) -> str:
    """Return the budget bucket for one kernel name."""
    if _has(name, _COLLECTIVE):
        return "allreduce"
    if "sm70_marlin_moe_gemm_kernel" in name:
        if _GATE_UP_TILE in name:
            return "moe_gate_up"
        if _DOWN_TILE in name:
            return "moe_down"
        return "moe_gemm_other"
    if "decode_moe_gemv_kernel" in name or "reduce_split_k_kernel" in name:
        return "moe_gate_up"
    if "fused_moe_kernel" in name:
        return "moe_draft"
    if _has(name, _MOE_ROUTER):
        return "moe_router"
    if _has(name, _MOE_SUPPORT):
        return "moe_align_reduce"
    if _has(name, _ATTENTION):
        return "attention"
    if _has(name, _NORM):
        return "norm"
    if _has(name, _DENSE_GEMM):
        return "dense_gemm"
    return "other"


def duration_stats(values: list[float]) -> dict:
    """Mean and tail of per-call durations, in microseconds."""
    ordered = sorted(values)
    n = len(ordered)

    def at(q: float) -> float:
        return ordered[min(n - 1, int(q * n))] if n else 0.0

    return {
        "count": n,
        "mean_us": round(sum(ordered) / n, 2) if n else 0.0,
        "p50_us": round(at(0.5), 2),
        "p90_us": round(at(0.9), 2),
        "p99_us": round(at(0.99), 2),
        "max_us": round(ordered[-1], 2) if n else 0.0,
    }


# The five buckets the issue asks for. Everything outside the first four lands
# in "norm/other"; the detail table splits that bucket back apart.
ROLLUP = {
    "moe_gate_up": "MoE gate_up",
    "moe_down": "MoE down",
    "attention": "attention",
    "allreduce": "allreduce",
}
ROLLUP_OTHER = "norm/other"


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        doc = json.load(fh)
    return doc.get("traceEvents", doc)


def kernel_events(events: list[dict]) -> list[dict]:
    return [
        ev
        for ev in events
        if ev.get("ph") == "X"
        and ev.get("cat") in ("kernel", "Kernel")
        and "dur" in ev
        and "name" in ev
    ]


def count_steps(kernels: list[dict], per_step: int) -> tuple[float, str]:
    """Return (steps, anchor name) from the most frequent anchor kernel."""
    counts: dict[str, int] = defaultdict(int)
    for ev in kernels:
        for anchor in ANCHOR_KERNELS:
            if anchor in ev["name"]:
                counts[anchor] += 1
    if not counts:
        raise SystemExit(
            "no step anchor in the trace; pass --steps N "
            f"(looked for {', '.join(ANCHOR_KERNELS)})"
        )
    name, count = max(counts.items(), key=lambda kv: kv[1])
    return count / per_step, name


def analyze(path: str, steps: float | None, per_step: int, topn: int) -> dict:
    kernels = kernel_events(load_events(path))
    if not kernels:
        raise SystemExit(f"no kernel events in {path}")

    anchor_name = ""
    if steps is None:
        steps, anchor_name = count_steps(kernels, per_step)
    if steps <= 0:
        raise SystemExit(f"step count must be positive, got {steps}")
    rounded = round(steps)
    if abs(steps - rounded) > 0.02:
        print(
            f"warning: {steps:.3f} forward steps from the anchor; "
            "the capture may include a partial step",
            file=sys.stderr,
        )
    steps = rounded or 1

    by_name: dict[str, list[float]] = {}
    durs: dict[str, list[float]] = defaultdict(list)
    for ev in kernels:
        rec = by_name.setdefault(ev["name"], [0.0, 0])
        rec[0] += float(ev["dur"])
        rec[1] += 1
        durs[ev["name"]].append(float(ev["dur"]))

    by_category: dict[str, float] = defaultdict(float)
    rows = []
    for name, (dur_us, count) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
        cat = classify(name)
        by_category[cat] += dur_us
        rows.append(
            {
                "kernel": name,
                "category": cat,
                "ms_per_step": dur_us / 1000.0 / steps,
                "us_per_call": dur_us / count,
                "calls_per_step": count / steps,
                "dur_us": dur_us,
                "stats": duration_stats(durs[name]),
            }
        )

    total_us = sum(v[0] for v in by_name.values())
    total_ms = total_us / 1000.0
    for row in rows:
        row["share"] = row["dur_us"] / total_us if total_us else 0.0
    # Wall span between the first and last kernel. The summed durations can
    # exceed it when streams overlap; the difference to the step's wall time
    # is the host gap, which the kernel budget does not cover.
    starts = [ev["ts"] for ev in kernels if "ts" in ev]
    span_ms = None
    first_ts = min(starts) if starts else None
    end_ts = None
    if starts:
        end_ts = max(ev["ts"] + float(ev["dur"]) for ev in kernels if "ts" in ev)
        span_ms = (end_ts - first_ts) / 1000.0 / steps

    # Where the slow collective calls sit: a first-step cluster is a prefill
    # tail in the window, not a steady-state cost. Keep the outliers, do not
    # trim them; the point is to read their position.
    ar_events = [
        ev for ev in kernels if classify(ev["name"]) == "allreduce" and "ts" in ev
    ]
    ar_outliers: list[dict] = []
    first_step_share = None
    if first_ts is not None and end_ts is not None and end_ts > first_ts:
        step_us = (end_ts - first_ts) / steps
        for ev in sorted(ar_events, key=lambda e: -float(e["dur"]))[:5]:
            ar_outliers.append(
                {
                    "us": round(float(ev["dur"]), 1),
                    "step": int((ev["ts"] - first_ts) / step_us),
                }
            )
        ar_total = sum(float(e["dur"]) for e in ar_events)
        if ar_total:
            first_step_share = (
                sum(float(e["dur"]) for e in ar_events if e["ts"] < first_ts + step_us)
                / ar_total
            )
    detail = {
        cat: {
            "ms_per_step": dur_us / 1000.0 / steps,
            "share": dur_us / total_us if total_us else 0.0,
        }
        for cat, dur_us in sorted(by_category.items(), key=lambda kv: -kv[1])
    }
    rollup: dict[str, float] = defaultdict(float)
    for cat, dur_us in by_category.items():
        rollup[ROLLUP.get(cat, ROLLUP_OTHER)] += dur_us

    result = {
        "trace": path,
        "steps": steps,
        "anchor_kernel": anchor_name,
        "kernel_ms_per_step": total_ms / steps,
        "span_ms_per_step": span_ms,
        "kernel_launches_per_step": len(kernels) / steps,
        "rollup": {
            label: {"ms_per_step": dur_us / 1000.0 / steps, "share": dur_us / total_us}
            for label, dur_us in sorted(rollup.items(), key=lambda kv: -kv[1])
        },
        "detail": detail,
        "allreduce": {
            "ms_per_step": by_category.get("allreduce", 0.0) / 1000.0 / steps,
            "share": by_category.get("allreduce", 0.0) / total_us if total_us else 0.0,
            "first_step_share": first_step_share,
            "outliers": ar_outliers,
            "kernels": [row for row in rows if row["category"] == "allreduce"],
        },
        "largest_kernel": rows[0] if rows else None,
        "kernels": rows[:topn] if topn else rows,
    }
    return result


def print_report(result: dict) -> None:
    print(
        f"trace: {result['trace']}\n"
        f"steps: {result['steps']}  anchor: {result['anchor_kernel'] or 'explicit'}\n"
        f"kernel time: {result['kernel_ms_per_step']:.2f} ms/step over "
        f"{result['kernel_launches_per_step']:.0f} launches/step"
        + (
            f"\nspan: {result['span_ms_per_step']:.2f} ms/step "
            f"(idle/gaps {result['span_ms_per_step'] - result['kernel_ms_per_step']:+.2f} ms)"
            if result.get("span_ms_per_step")
            else ""
        )
        + "\n"
    )
    print(f"{'buckets':<14} {'ms/step':>9} {'share':>7}")
    for label, rec in result["rollup"].items():
        print(f"{label:<14} {rec['ms_per_step']:9.3f} {rec['share']*100:6.1f}%")
    print(f"\n{'detail':<18} {'ms/step':>9} {'share':>7}")
    for cat, rec in result["detail"].items():
        print(f"{cat:<18} {rec['ms_per_step']:9.3f} {rec['share']*100:6.1f}%")

    ar = result["allreduce"]
    print(
        f"\nallreduce: {ar['ms_per_step']:.3f} ms/step, "
        f"{ar['share']*100:.1f}% of kernel time"
    )
    for row in ar["kernels"][:5]:
        st = row["stats"]
        print(
            f"  {row['ms_per_step']:7.3f} ms/step {row['share']*100:5.1f}% "
            f"{row['calls_per_step']:7.1f} calls/step {row['us_per_call']:6.1f} us/call "
            f"(p50 {st['p50_us']:.1f} p90 {st['p90_us']:.1f} "
            f"p99 {st['p99_us']:.1f} max {st['max_us']:.0f})  {row['kernel'][:70]}"
        )
    largest = result.get("largest_kernel")
    if largest:
        print(
            f"largest single kernel: {largest['ms_per_step']:.3f} ms/step "
            f"({largest['share']*100:.1f}%) [{largest['category']}] "
            f"{largest['kernel'][:70]}"
        )
    if ar.get("first_step_share") is not None:
        print(f"  allreduce first-step share: {ar['first_step_share']*100:.1f}%")
    if ar.get("outliers"):
        spots = ", ".join(f"{o['us']:.1f}us@step{o['step']}" for o in ar["outliers"])
        print(f"  slowest allreduce calls: {spots}")

    print(f"\n{'ms/step':>9} {'calls/st':>9} {'us/call':>9}  category / kernel")
    for row in result["kernels"][:20]:
        print(
            f"{row['ms_per_step']:9.3f} {row['calls_per_step']:9.1f} "
            f"{row['us_per_call']:9.1f}  {row['category']}: {row['kernel'][:80]}"
        )


# --- collection ------------------------------------------------------------
def build_exact(tok, n_tokens: int, seed: str) -> tuple[str, int]:
    """Return (text, actual) for a prompt of exactly ``n_tokens`` tokens.

    Repeated token blocks do not round-trip through text unchanged, so the
    candidate is retargeted by the shortfall instead of trusting a ratio.
    """
    ids = tok.encode(seed, add_special_tokens=False)
    if not ids:
        raise SystemExit("seed text produced no tokens")
    if n_tokens < len(ids):
        raise SystemExit(f"asked for {n_tokens} tokens, the seed alone is {len(ids)}")
    want = n_tokens
    got = None
    for _ in range(8):
        reps = (want + len(ids) - 1) // len(ids)
        text = tok.decode((ids * reps)[:want])
        got = len(tok(text, add_special_tokens=False).input_ids)
        if got == n_tokens:
            return text, got
        want += n_tokens - got
    raise SystemExit(f"could not build a {n_tokens}-token prompt after 8 passes (last {got})")


def post(base: str, path: str, payload: dict, timeout: float = 3600.0) -> str:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def stream_request(base: str, payload: dict, timeout: float = 3600.0) -> dict:
    """Send one streamed completion and return its timing and token counts."""
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_t = last_t = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    last_t = now
    end_t = time.perf_counter()
    return {"t0": t0, "first_t": first_t, "last_t": last_t, "end_t": end_t, "usage": usage}


def anchor_rate(base: str, payload: dict) -> dict:
    r = stream_request(base, payload)
    if r["first_t"] is None or r["last_t"] is None:
        return {"error": "no output tokens"}
    out = (r["usage"] or {}).get("completion_tokens")
    if not out:
        return {"error": "no usage.completion_tokens"}
    dec = r["last_t"] - r["first_t"]
    return {
        "output_tokens": out,
        "ttft_s": round(r["first_t"] - r["t0"], 3),
        "decode_s": round(dec, 3),
        "pure_decode_tps": round(max(out - 1, 0) / dec, 2) if dec > 0 else None,
    }


def collect(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    base = "http://" + args.host
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text, plen = build_exact(tok, args.context_tokens, args.seed)
    print(f"prompt tokens: {plen}", flush=True)

    # Use the served id: some launches set it from the checkpoint path, not
    # from the repository's short name.
    model = args.model
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
            ids = [m["id"] for m in json.load(resp).get("data", [])]
        if ids:
            model = ids[0]
    except Exception:  # the endpoint may not answer until the first request
        pass
    print(f"model: {model}", flush=True)

    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": args.output_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 0,
    }
    # Warm the prefix so the profiled request starts decoding under 2 s and
    # the capture window cannot absorb a prefill.
    warm = dict(payload, max_tokens=4)
    stream_request(base, warm)
    print("warmup done", flush=True)

    anchor = anchor_rate(base, payload)
    print("anchor: " + json.dumps(anchor), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.out_dir, "**", "*.json*"), recursive=True))

    def send() -> None:
        stream_request(base, payload)

    worker = threading.Thread(target=send)
    worker.start()
    time.sleep(args.arm_after)
    msg = post(
        base,
        "/start_profile",
        {"num_steps": args.steps, "activities": ["GPU"], "output_dir": args.out_dir},
    )
    print("profile: " + msg.strip(), flush=True)
    worker.join()
    print("request done", flush=True)
    time.sleep(args.flush_wait)

    files = [
        f
        for f in glob.glob(os.path.join(args.out_dir, "**", "*.json*"), recursive=True)
        if f not in before and f.endswith((".json", ".json.gz"))
    ]
    if not files:
        raise SystemExit(f"no trace written under {args.out_dir}")
    # One file per TP rank; rank 0 is representative for kernel time.
    rank0 = [f for f in files if "-TP-0." in os.path.basename(f)]
    path = max(rank0 or files, key=os.path.getmtime)
    result = analyze(path, args.explicit_steps, args.per_step_anchors, args.topn)
    result["anchor"] = anchor
    result["prompt_tokens"] = plen
    result["output_tokens"] = args.output_tokens
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("collect", help="measure the anchor, capture a decode window, analyze")
    c.add_argument("--host", default="127.0.0.1:11436")
    c.add_argument("--model", default="qwen38-flashnext-nvfp4")
    c.add_argument("--model-dir", required=True, help="checkpoint for the tokenizer")
    c.add_argument("--out-dir", default="/tmp/prof-decode")
    c.add_argument("--context-tokens", type=int, default=4096)
    c.add_argument("--output-tokens", type=int, default=400)
    c.add_argument("--steps", type=int, default=16, help="forward steps to capture")
    c.add_argument("--arm-after", type=float, default=2.0)
    c.add_argument("--flush-wait", type=float, default=5.0)
    c.add_argument("--per-step-anchors", type=int, default=DEFAULT_PER_STEP_ANCHORS)
    c.add_argument(
        "--explicit-steps",
        type=float,
        default=None,
        help="skip anchor counting and divide by this many steps",
    )
    c.add_argument("--topn", type=int, default=40)
    c.add_argument(
        "--seed",
        default=(
            "The measure of a language model is not the speed of one request but "
            "the behaviour of the system under load, over time, and across contexts. "
        ),
    )
    c.add_argument("--json-out", default=None)

    a = sub.add_parser("analyze", help="aggregate an existing trace")
    a.add_argument("trace")
    a.add_argument("--steps", type=float, default=None)
    a.add_argument("--per-step-anchors", type=int, default=DEFAULT_PER_STEP_ANCHORS)
    a.add_argument("--topn", type=int, default=40)
    a.add_argument("--json-out", default=None)

    args = ap.parse_args()
    if args.mode == "collect":
        result = collect(args)
    else:
        result = analyze(args.trace, args.steps, args.per_step_anchors, args.topn)

    print_report(result)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
