"""CPU tests for the decode-budget trace classifier and aggregator."""

from __future__ import annotations

import gzip
import http.server
import importlib.util
import json
import pathlib
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "trace_decode_budget", ROOT / "scripts" / "trace_decode_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget = _load()
analyze = budget.analyze
classify = budget.classify


def test_classify_names_the_requested_buckets():
    assert classify("void sglang::all_reduce_1shot_push_kernel<...>") == "allreduce"
    assert classify("ncclDevKernel_AllReduce_Sum_f16_RING_LL") == "allreduce"
    assert classify("decode_moe_gemv_kernel") == "moe_gate_up"
    assert classify("reduce_split_k_kernel") == "moe_gate_up"
    assert classify(
        "void marlin_moe_wna16::sm70_marlin_moe_gemm_kernel<..., "
        "Sm70MoeNvfp4GemmSpec, 64, 64, 32, ...>"
    ) == "moe_gate_up"
    assert classify(
        "void marlin_moe_wna16::sm70_marlin_moe_gemm_kernel<..., "
        "Sm70MoeNvfp4GemmSpec, 32, 256, 32, ...>"
    ) == "moe_down"
    assert classify("fused_moe_kernel") == "moe_draft"
    assert classify("_router_triton_kernel") == "moe_router"
    assert classify("_moe_align_small_numel_kernel") == "moe_align_reduce"
    assert classify("sm70_longctx::qsa_decode_partial_kernel<...>") == "attention"
    assert classify("grouped_gemma_rmsnorm_kernel<...>") == "norm"
    assert classify("turbomind::gemm::gemm_kernel<...>") == "dense_gemm"
    assert classify("gather_ple_embedding_sm70") == "other"


def _event(name, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "dur": dur, "ts": 0}


def test_analyze_divides_by_the_anchor_and_rolls_up(tmp_path):
    events = [_event("_moe_align_small_numel_kernel", 10.0) for _ in range(51)]
    events += [
        _event("decode_moe_gemv_kernel", 100.0),
        _event("sm70_marlin_moe_gemm_kernel<Sm70MoeNvfp4GemmSpec, 32, 256, 32>", 40.0),
        _event("sm70_longctx::qsa_decode_partial_kernel<__half, 6>", 30.0),
        _event("all_reduce_1shot_push_kernel", 20.0),
        _event("grouped_gemma_rmsnorm_kernel<2560l, false, __half>", 5.0),
        _event("ignored: not a kernel", 999.0, cat="cpu_op"),
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)

    result = analyze(str(path), steps=None, per_step=51, topn=None)

    assert result["steps"] == 1
    assert result["anchor_kernel"] == "_moe_align_small_numel_kernel"
    # 51 * 10 us of align + 195 us of the named kernels, per one forward.
    assert result["kernel_ms_per_step"] == 705.0 / 1000.0
    assert result["rollup"]["MoE gate_up"]["ms_per_step"] == 0.1
    assert result["rollup"]["MoE down"]["ms_per_step"] == 0.04
    assert result["rollup"]["attention"]["ms_per_step"] == 0.03
    assert result["rollup"]["allreduce"]["ms_per_step"] == 0.02
    assert result["rollup"]["norm/other"]["ms_per_step"] == 0.515
    assert result["allreduce"]["ms_per_step"] == 0.02
    assert [row["kernel"] for row in result["allreduce"]["kernels"]] == [
        "all_reduce_1shot_push_kernel"
    ]
    assert result["largest_kernel"]["category"] == "moe_align_reduce"


def test_duration_stats_reports_the_tail():
    st = budget.duration_stats([1.0, 2.0, 3.0, 100.0])
    assert st["count"] == 4
    assert st["mean_us"] == 26.5
    assert st["p50_us"] == 3.0
    assert st["max_us"] == 100.0


def test_allreduce_outliers_keep_their_step(tmp_path):
    events = []
    for step in range(2):
        for i in range(51):
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "_moe_align_small_numel_kernel",
                    "dur": 1.0,
                    "ts": step * 1000 + i,
                }
            )
    events += [
        {"ph": "X", "cat": "kernel", "name": "all_reduce_1shot_push_kernel",
         "dur": 300.0, "ts": 0.0},
        {"ph": "X", "cat": "kernel", "name": "all_reduce_1shot_push_kernel",
         "dur": 10.0, "ts": 1500.0},
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)

    result = analyze(str(path), steps=None, per_step=51, topn=None)

    assert result["steps"] == 2
    assert result["allreduce"]["outliers"][0] == {"us": 300.0, "step": 0}
    assert result["allreduce"]["first_step_share"] == 300.0 / 310.0


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server's spelling
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(
            b"data: {\"choices\": [{\"text\": \"a\"}], \"usage\": null}\n\n"
            b"data: {\"choices\": [{\"text\": \"b\"}], "
            b"\"usage\": {\"prompt_tokens\": 4, \"completion_tokens\": 2}}\n\n"
            b"data: [DONE]\n\n"
        )

    def log_message(self, *args):
        pass


def test_stream_request_reads_usage_and_token_span():
    server = http.server.HTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        raw = budget.stream_request(base, {"model": "m", "prompt": "p"})
        assert raw["usage"]["completion_tokens"] == 2
        assert raw["first_t"] is not None
        assert raw["last_t"] >= raw["first_t"]
        rate = budget.anchor_rate(base, {"model": "m", "prompt": "p"})
        assert rate["output_tokens"] == 2
        assert rate["pure_decode_tps"] > 0
    finally:
        server.shutdown()
        server.server_close()
