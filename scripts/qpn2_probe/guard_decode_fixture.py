"""Decode-only P2 guard on 6a's deterministic fixture (#224).

Compares the QPN2 decode path against the Marlin control on the same bytes,
through the shipping entry point ``moe_wna16_marlin_gemm``. The arm matrix is
``marlin`` / ``qpn2 both`` / ``qpn2 w13`` / ``qpn2 w2``; each arm prepares the
layer with ``prepare_moe_nvfp4`` and then issues the real call.

The fixture is built by ``fixture_qpn2.py`` from branch
``review/qpn2-oracle-224`` (not vendored here). Put ``fixture_qpn2.py`` and
``oracle_qpn2.py`` next to this script, or point ``--fixture-src`` at them.

Pass: ``max|qpn2 - marlin| / max|expected| <= 1e-3`` and
``max|qpn2 - expected| / max|expected| <= 1e-3`` for every shape and arm,
where ``expected`` is the fixture's FP32 reference.

Usage::

    python scripts/qpn2_probe/guard_decode_fixture.py --fixture-dir /tmp/qpn2_fixture_v1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SHAPES = {"w13": (320, 2560), "w2": (2560, 160)}
BLOCK = 8
TOLERANCE = 1e-3
LAYOUT_ENV = "SGLANG_SM70_FP4_MOE_LAYOUT"
TENSORS_ENV = "SGLANG_SM70_QPN2_TENSORS"


def _fixture_module(src: Path):
    sys.path.insert(0, str(src))
    import fixture_qpn2  # noqa: PLC0415 - path-dependent import

    return fixture_qpn2


def _bank(torch, fixture, shape: str, device):
    case = fixture[shape]
    n, k = SHAPES[shape]
    weight = torch.from_numpy(obytes(case["codes"])).to(device)
    scale_bytes = torch.from_numpy(obytes(case["scale_bytes"]))
    scales = scale_bytes.view(torch.float8_e4m3fn).to(device)
    gscale = torch.from_numpy(obytes(case["global_scale"])).float().to(device)
    assert tuple(weight.shape) == (weight.shape[0], n, k // 2), weight.shape
    assert tuple(scales.shape) == (weight.shape[0], n, k // 16), scales.shape
    return weight, scales, gscale


def obytes(array):
    import numpy as np

    return np.ascontiguousarray(array)


def _make_layer(torch, bank):
    layer = SimpleNamespace(
        params_dtype=torch.float16,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_config=SimpleNamespace(group_size=16),
        w13_bias=None,
        w2_bias=None,
    )
    for prefix, (weight, scales, gscale) in bank.items():
        setattr(layer, prefix + "_weight", weight.clone())
        setattr(layer, prefix + "_weight_scale", scales.clone())
        setattr(layer, prefix + "_weight_scale_2", gscale.clone())
    return layer


def _prepare(torch, fixture, layout: str, tensors: str, device):
    from sglang_restomod.quant.sm70_fp4_moe.method import prepare_moe_nvfp4

    bank = {prefix: _bank(torch, fixture, prefix, device) for prefix in SHAPES}
    layer = _make_layer(torch, bank)
    previous = (os.environ.get(LAYOUT_ENV), os.environ.get(TENSORS_ENV))
    os.environ[LAYOUT_ENV] = layout
    os.environ[TENSORS_ENV] = tensors
    try:
        prepare_moe_nvfp4(layer)
    finally:
        for key, value in zip((LAYOUT_ENV, TENSORS_ENV), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return layer


def _routing(torch, fixture, shape: str, pair: bool, device):
    case = fixture[shape]
    sentinel = int(fixture.manifest["shapes"][shape]["sentinel_value"])
    if pair:
        # Two routes of expert 0 in one block (token 0's two top-k slots).
        # Both kernels address output rows by route index, so the two slots
        # must carry distinct indices; a repeated index leaves a row unwritten.
        sorted_ids = torch.from_numpy(obytes(case["pair_sorted_token_ids"])).int().to(device)
        real = [int(v) for v in sorted_ids.tolist() if int(v) != sentinel]
        assert len(real) == len(set(real)), f"pair route indices alias: {real}"
        return dict(
            sorted_token_ids=sorted_ids,
            expert_ids=torch.from_numpy(obytes(case["pair_expert_ids"])).int().to(device),
            padded=torch.from_numpy(obytes(case["pair_num_tokens_post_padded"])).int().to(device),
            topk_weights=torch.from_numpy(obytes(case["pair_topk_weights"])).float().to(device),
            top_k=2,
            rows=2,
            size_m=1,
        )
    return dict(
        sorted_token_ids=torch.from_numpy(obytes(case["sorted_token_ids"])).int().to(device),
        expert_ids=torch.from_numpy(obytes(case["expert_ids"])).int().to(device),
        padded=torch.from_numpy(obytes(case["num_tokens_post_padded"])).int().to(device),
        topk_weights=torch.from_numpy(obytes(case["topk_weights"])).float().to(device),
        top_k=1,
        rows=int(case["expert_ids"].shape[0]),
        size_m=int(case["expert_ids"].shape[0]),
    )


def _run(torch, layer, prefix, x, routing, size_n, size_k, force_marlin):
    from sgl_kernel.scalar_type import scalar_types

    from sglang_restomod.quant.sm70_fp4_moe.method import (
        MARLIN_SPLIT_K,
        moe_wna16_marlin_gemm,
    )

    previous = (os.environ.get("SM70_MOE_DECODE_GEMV"), os.environ.get(MARLIN_SPLIT_K))
    if force_marlin:
        os.environ["SM70_MOE_DECODE_GEMV"] = "0"
    try:
        out = torch.empty(routing["rows"], size_n, dtype=torch.float16, device=x.device)
        moe_wna16_marlin_gemm(
            x, out, getattr(layer, prefix + "_weight"), None,
            getattr(layer, prefix + "_weight_scale"),
            getattr(layer, prefix + "_weight_scale_2"),
            None, None, None, torch.empty(0, dtype=torch.int32, device=x.device),
            routing["sorted_token_ids"], routing["expert_ids"], routing["padded"],
            routing["topk_weights"], BLOCK, routing["top_k"], False, False,
            scalar_types.float4_e2m1f, routing["size_m"], size_n, size_k,
        )
        return out
    finally:
        for key, value in zip(("SM70_MOE_DECODE_GEMV", MARLIN_SPLIT_K), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _rel(a, b, scale):
    return float((a.float() - b.float()).abs().max()) / scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--fixture-src", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    fixture_module = _fixture_module(Path(args.fixture_src))
    fixture = fixture_module.load(args.fixture_dir)
    device = torch.device(args.device)

    arms = {"marlin": _prepare(torch, fixture, "marlin", "both", device)}
    for tensors in ("both", "w13", "w2"):
        arms[f"qpn2-{tensors}"] = _prepare(torch, fixture, "qpn2", tensors, device)

    print(f"fixture {args.fixture_dir}  {torch.cuda.get_device_name(0)}")
    failures = 0
    for prefix, (size_n, size_k) in SHAPES.items():
        reference = fixture.reference(prefix)["expected"]
        expected = torch.from_numpy(obytes(reference)).to(device)
        scale = expected.abs().max().item()
        x = torch.from_numpy(obytes(fixture[prefix]["activation"])).to(device)
        for pair in (False, True):
            routing = _routing(torch, fixture, prefix, pair, device)
            if routing["top_k"] != 1:
                # Both routes are expert 0 and token 0: each route row is the
                # expert-0 dot product.
                expected = expected[:1].expand(routing["rows"], -1).contiguous()
                # One token, two route rows: the kernel recovers the token
                # from the route index, and `a` is [size_m, K].
                x = x[:1].contiguous()
            base = None
            print(f"\n{prefix} {'pair (2 routes, 1 block)' if pair else 'single route per block'}"
                  f"  n={size_n} k={size_k} routes={routing['rows']}")
            print(f"  {'arm':<12} {'vs expected':>12} {'vs marlin':>12}")
            for name, layer in arms.items():
                out = _run(torch, layer, prefix, x, routing, size_n, size_k,
                           force_marlin=(name == "marlin"))
                err = _rel(out, expected, scale)
                if name == "marlin":
                    base = out
                pairwise = 0.0 if base is None else _rel(out, base, scale)
                flag = ""
                if name != "marlin" and (err > TOLERANCE or pairwise > TOLERANCE):
                    flag = "  FAIL"
                    failures += 1
                print(f"  {name:<12} {err:12.3e} {pairwise:12.3e}{flag}")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} arm(s) over {TOLERANCE:.0e}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
