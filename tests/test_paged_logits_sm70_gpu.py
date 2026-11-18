"""GPU gate for the paged logits kernel: the wired band must clear 2x, exactly.

Marked ``gpu``: it needs a V100 and the pin's indexer module. The full gate and
its evidence are in `docs/runs/2026-11-17-indexer-kernel-first-run.md`; this is
the regression-sized version — B=1 and the wired edge, captured, agreement and
top-64 set equality included.
"""
from __future__ import annotations

import statistics

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

H, D, BLK = 32, 128, 64


def _build(b, pages, seq_len, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(b, 1, H, D, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    kv = torch.zeros(pages, BLK * (D + 4), dtype=torch.uint8, device="cuda")
    kv[:, : BLK * D] = (
        torch.randn(pages, BLK * D, device="cuda", generator=g) * 0.5
    ).to(torch.float8_e4m3fn).view(torch.uint8)
    kv[:, BLK * D:] = (
        torch.rand(pages, BLK, device="cuda", generator=g) * 0.02
    ).view(torch.float32).view(torch.uint8).view(pages, BLK * 4)
    weight = torch.rand(b, H, device="cuda", generator=g)
    seq = torch.full((b,), seq_len, dtype=torch.int32, device="cuda")
    pt = torch.randint(0, pages, (b, pages), dtype=torch.int32, device="cuda", generator=g)
    return q, kv.view(pages, BLK, 1, D + 4), weight, seq, pt


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _median_replay(graph, warmup=10, reps=50):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); graph.replay(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def test_kernel_clears_the_wired_band():
    from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch as ref
    from sglang_restomod.kernels.deep_gemm_sm70 import PAGED_LOGITS_AUTO_MAX_B
    from sglang_restomod.kernels.paged_logits_sm70.kernel import fused_paged_mqa_logits

    for b in (1, PAGED_LOGITS_AUTO_MAX_B):
        q, kv, w, seq, pt = _build(b, 128, 8192)
        ref_fn = lambda: ref(q, kv, w, seq, pt, None, 8192, False)
        ker_fn = lambda: fused_paged_mqa_logits(q, kv, w, seq, pt, None, 8192, False)
        want, got = ref_fn(), ker_fn()
        torch.cuda.synchronize()

        assert (got - want).abs().max().item() <= 2e-2
        assert torch.equal(
            got.topk(64, dim=1).indices.sort(dim=1).values,
            want.topk(64, dim=1).indices.sort(dim=1).values,
        ), f"top-64 set differs at B={b}"

        g_ref, _ = _capture(ref_fn)
        g_ker, _ = _capture(ker_fn)
        ratio = _median_replay(g_ref) / _median_replay(g_ker)
        assert ratio >= 2.0, f"B={b}: {ratio:.2f}x against the 2.0x band"
