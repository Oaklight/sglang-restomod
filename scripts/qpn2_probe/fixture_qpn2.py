#!/usr/bin/env python3
"""Deterministic P2 fixture for the QPN2 decode guard (#224).

Builds, on CPU, one 8-expert NVFP4 bank at the production per-rank shapes
(``w13`` N=320 K=2560, ``w2`` N=2560 K=160) in the checkpoint format that both
consumers read, plus the QPN2 bytes for the same codes and scales. The point is
that every arm of the guard loads the same bytes instead of building its own
random bank, so a difference is the kernel, not the input.

Determinism does not come from a torch generator: torch makes no cross-version
promise about ``randint``/``randn`` streams. Every input tensor here is a
prefix of a SHAKE-256 stream keyed by the fixture version and the tensor name,
so the bytes are identical on any host and any torch build. ``manifest.json``
carries the SHA-256 of every stored tensor; ``verify`` rebuilds from the
streams and compares.

Layout: ``manifest.json`` + ``tensors.npz`` in the output directory; numpy only,
so the fixture loads without torch. The dense reference and the expected expert
outputs are derived, not stored: they are
``code * fp16(e4m3(scale_byte) * global_scale)`` (one FP16 RNE on the scale,
the spec path) and the activation times that dense weight, both accumulated in
float64 and cast to float32. Routing is emitted in the shipping layout
(``sorted_token_ids``, ``expert_ids``, ``num_tokens_post_padded``, block 8), with
one route per block and a second case of two routes of the same expert in one
block.

Marlin layout: ``build --with-marlin`` runs once on a GPU host and calls the
shipping ``_repack_experts`` / ``encode_scales`` / ``encode_global_scale``, so
the stored Marlin bytes come from the shipping path. There is deliberately no
CPU reimplementation of ``gptq_marlin_repack``: the Marlin arm is the control,
and a second implementation would be a second thing that can be wrong.

Usage::

    python scripts/qpn2_probe/fixture_qpn2.py build --out /tmp/qpn2_fixture_v1
    python scripts/qpn2_probe/fixture_qpn2.py verify /tmp/qpn2_fixture_v1
    # on a V100 host with the extension built:
    SM70_MARLIN_DIR=... python .../fixture_qpn2.py build --out DIR --with-marlin

Consumer::

    from fixture_qpn2 import load
    fx = load("/tmp/qpn2_fixture_v1")
    case = fx["w13"]                 # codes, scale_bytes, global_scale, qpn2, ...
    ref = fx.reference("w13")        # dense [E, N, K] fp32, expected [E, N] fp32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_qpn2  # noqa: E402 - sibling module, stdlib only

VERSION = "qpn2-fixture-v1"
EXPERTS = 8
ROUTE_ROWS = 8  # one route row per expert, as the decode align kernel emits
MOE_BLOCK = 8  # the decode align kernel's route tile
SENTINEL = ROUTE_ROWS  # padded-slot value; also the route count

# (n, k) per shape, production per-rank values at moe_intermediate_size=640,
# TP4: w13 = [2*inter, hidden], w2 = [hidden, inter].
SHAPES = {
    "w13": (320, 2560),
    "w2": (2560, 160),
}

E2M1 = np.array(oracle_qpn2.E2M1, dtype=np.float64)


def _stream(name: str, nbytes: int) -> np.ndarray:
    return np.frombuffer(
        hashlib.shake_256(f"{VERSION}|{name}".encode()).digest(nbytes), dtype=np.uint8
    )


def _codes(shape: str, experts: int, n: int, k: int) -> np.ndarray:
    """[E, N, K/2] packed bytes; low nibble is even K, the ModelOpt order."""
    nibbles = (_stream(f"{shape}|codes", experts * n * k) & 0xF).reshape(
        experts, n, k // 2, 2
    )
    return (nibbles[..., 0] | (nibbles[..., 1] << 4)).astype(np.uint8)


def _scale_bytes(shape: str, experts: int, n: int, k: int) -> np.ndarray:
    """E4M3 bytes with exponents 6..8 (0.5..8), the range a real NVFP4 group
    scale lands in after ModelOpt normalizes by the global scale."""
    raw = _stream(f"{shape}|scales", experts * n * (k // 16))
    mantissa = raw & 7
    exponent = (6 + ((raw >> 3) % 3)).astype(np.uint8)
    return (mantissa | (exponent << 3)).reshape(experts, n, k // 16)


def _global_scale(shape: str, experts: int) -> np.ndarray:
    raw = _stream(f"{shape}|global_scale", experts).astype(np.float64)
    return (0.0123 * (1.0 + 0.1 * raw / 255.0)).astype(np.float32)


def _activations(shape: str, rows: int, k: int) -> np.ndarray:
    """FP16 [rows, k]. Every value is 1.mmm * 2**e with a 10-bit mantissa, so it
    is exact in FP16; nothing is rounded on the way in."""
    raw = _stream(f"{shape}|activations", rows * k * 2).view(np.uint16)
    sign = np.where((raw >> 15) & 1, -1.0, 1.0)
    exponent = ((raw >> 13) & 3).astype(np.float64) - 2.0  # 2**-2 .. 2**1
    mantissa = (raw & 0x3FF).astype(np.float64) / 1024.0
    values = sign * np.exp2(exponent) * (1.0 + mantissa)
    return values.astype(np.float16).reshape(rows, k)


def e4m3_to_float(bytes_: np.ndarray) -> np.ndarray:
    """Exact E4M3 byte -> float64, by hand (no CPU float8 indexing)."""
    b = bytes_.astype(np.uint8)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exponent = (b >> 3) & 0xF
    mantissa = (b & 7).astype(np.float64)
    subnormal = sign * mantissa * 2.0**-9
    normal = sign * np.exp2(exponent.astype(np.float64) - 7.0) * (1.0 + mantissa / 8.0)
    return np.where(exponent == 0, subnormal, normal)


def dense_reference(codes: np.ndarray, scale_bytes: np.ndarray, global_scale: float) -> np.ndarray:
    """[N, K/2] packed + [N, K/16] E4M3 bytes -> FP32 [N, K], spec scale path.

    One rounding: ``fp16(e4m3 * global_scale)``. The code product is exact in
    float64 and cast once at the end.
    """
    n = codes.shape[0]
    k = codes.shape[1] * 2
    nib = codes.astype(np.uint16)
    led = np.stack([nib & 0xF, nib >> 4], axis=-1).reshape(n, k).astype(np.int64)
    magnitude = E2M1[led & 7]
    values = np.where((led & 8) != 0, -magnitude, magnitude)
    scale = (e4m3_to_float(scale_bytes) * np.float64(global_scale)).astype(np.float16)
    return (values * np.repeat(scale.astype(np.float64), 16, axis=1)).astype(np.float32)


def qpn2_bytes(codes: np.ndarray, scale_bytes: np.ndarray, n: int, k: int) -> np.ndarray:
    """Per-expert QPN2 bytes from the independent oracle (spec section 4)."""
    return np.frombuffer(
        oracle_qpn2.oracle_pack(codes.tolist(), scale_bytes.tolist(), n, k), dtype=np.uint8
    ).copy()


def _tensor_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _manifest_entry(array: np.ndarray) -> dict:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _tensor_digest(array),
    }


def _import_torch():
    import torch

    return torch


def build(out: Path, experts: int, with_marlin: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    manifest = {
        "version": VERSION,
        "experts": experts,
        "route_rows": ROUTE_ROWS,
        "shapes": {},
        "reference": {
            "codes": "ModelOpt packed uint8 [E, N, K/2], low nibble = even K",
            "scale_bytes": "E4M3 [E, N, K/16], one FP16 RNE on e4m3*global_scale",
            "dense": "code * fp16(e4m3(byte) * global_scale), accumulated in float64",
            "expected": "dense[E] @ activation row, float64 then float32",
        },
        "producer": "oracle_qpn2.oracle_pack (spec section 4)",
    }

    for shape, (n, k) in SHAPES.items():
        codes = _codes(shape, experts, n, k)
        scales = _scale_bytes(shape, experts, n, k)
        gs = _global_scale(shape, experts)
        act = _activations(shape, ROUTE_ROWS, k)

        packed_qpn2 = np.stack([qpn2_bytes(codes[e], scales[e], n, k) for e in range(experts)])
        dense = np.stack(
            [dense_reference(codes[e], scales[e], float(gs[e])) for e in range(experts)]
        )

        # The decode must be bit-identical to the dense reference: both use the
        # same scale path, so any difference is a permutation bug in the pack.
        for e in range(experts):
            decoded = np.array(
                oracle_qpn2.oracle_decode(
                    packed_qpn2[e].tolist(), n, k, float(gs[e]), fp16_scale=True
                ),
                dtype=np.float32,
            )
            diff = float(np.abs(decoded - dense[e]).max())
            if diff != 0.0:
                raise SystemExit(f"{shape} expert {e}: QPN2 decode differs from dense by {diff}")

        expected = np.empty((experts, n), dtype=np.float32)
        for e in range(experts):
            expected[e] = (dense[e].astype(np.float64) @ act[e].astype(np.float64)).astype(
                np.float32
            )

        # Routing in the shipping layout: block r carries route r in slot 0 and
        # the sentinel in the other slots. The second case is the general path
        # the probe does not cover: one block, two routes of the same expert
        # (token 0 in both top-k slots), so its output is the route sum.
        sorted_ids = np.full((experts, MOE_BLOCK), SENTINEL, dtype=np.int32)
        sorted_ids[np.arange(experts), 0] = np.arange(experts, dtype=np.int32)
        pair_sorted_ids = np.full((MOE_BLOCK,), SENTINEL, dtype=np.int32)
        pair_sorted_ids[:2] = 0
        pair_expected = (
            2.0 * (dense[0].astype(np.float64) @ act[0].astype(np.float64))
        ).astype(np.float32)

        entry = {
            "n": n,
            "k": k,
            "tiles": (n + 31) // 32,
            "groups": k // 16,
            "qpn2_bytes_per_expert": ((n + 31) // 32) * (k // 16) * 288,
            "top_k": 1,
            "moe_block_size": MOE_BLOCK,
            "sentinel_value": SENTINEL,
            "pair_top_k": 2,
            # Recorded so the harness labels the arm it actually ran: the
            # shipping path serves w13 from decode_gemv (N/64 odd) and w2 from
            # Marlin (N/64 even) at M <= 8.
            "shipping_dispatch": (
                "decode_gemv" if n % 64 == 0 and (n // 64) % 2 == 1 else "marlin"
            ),
            "dense_sha256": _tensor_digest(dense),
        }

        named = {
            "codes": codes,
            "scale_bytes": scales,
            "global_scale": gs,
            "qpn2": packed_qpn2,
            "activation": act,
            "expert_ids": np.arange(ROUTE_ROWS, dtype=np.int32),
            "row_token_ids": np.arange(ROUTE_ROWS, dtype=np.int32),
            "topk_weights": np.ones(ROUTE_ROWS, dtype=np.float32),
            "sorted_token_ids": sorted_ids.reshape(-1),
            "num_tokens_post_padded": np.array([experts * MOE_BLOCK], dtype=np.int32),
            "pair_sorted_token_ids": pair_sorted_ids,
            "pair_expert_ids": np.zeros(1, dtype=np.int32),
            "pair_num_tokens_post_padded": np.array([MOE_BLOCK], dtype=np.int32),
            "pair_topk_weights": np.ones(2, dtype=np.float32),
            "pair_expected": pair_expected,
            "expected": expected,
        }
        for key, value in named.items():
            tensors[f"{shape}.{key}"] = value
        entry["tensors"] = {key: _manifest_entry(value) for key, value in named.items()}
        manifest["shapes"][shape] = entry

    if with_marlin:
        _build_marlin(tensors, manifest, experts)

    np.savez(out / "tensors.npz", **tensors)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(v.nbytes for v in tensors.values())
    print(f"wrote {out} ({len(tensors)} tensors, {total / 1e6:.1f} MB)")


def _build_marlin(tensors: dict, manifest: dict, experts: int) -> None:
    """Run the shipping conversion once and store its output as data.

    Needs a GPU and ``SM70_MARLIN_DIR``; the fixture then loads without either.
    """
    import glob

    torch = _import_torch()
    from sglang_restomod.quant.sm70_fp4_moe.marlin import directory
    from sglang_restomod.quant.sm70_fp4_moe.method import _repack_experts
    from sglang_restomod.quant.sm70_fp4_moe.scales import (
        encode_global_scale,
        encode_scales,
    )

    device = torch.device("cuda")
    info = {"repack": "_repack_experts (gptq_marlin_repack)", "scales": "encode_scales fp16"}
    for shape, (n, k) in SHAPES.items():
        codes = torch.from_numpy(tensors[f"{shape}.codes"]).to(device)
        repped = _repack_experts(codes).cpu().numpy()  # int32 [E, K/16, N*2]
        scales = (
            torch.from_numpy(tensors[f"{shape}.scale_bytes"].copy())
            .view(torch.float8_e4m3fn)
            .to(device)
        )
        encoded, factor = encode_scales(scales.transpose(1, 2).contiguous(), torch.float16)
        gs = torch.from_numpy(tensors[f"{shape}.global_scale"]).to(device)
        gs_eff = (encode_global_scale(gs, torch.float16) / factor).cpu().numpy()

        for key, value in {
            "marlin_weight": repped,
            "marlin_scales": encoded.view(torch.uint8).cpu().numpy(),
            "marlin_global_scale": gs_eff.astype(np.float32),
        }.items():
            tensors[f"{shape}.{key}"] = value
            manifest["shapes"][shape]["tensors"][key] = _manifest_entry(value)
    info["extension_dir"] = str(directory())
    info["libraries"] = {
        Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in sorted(glob.glob(str(directory() / "_*C*.so")))
    }
    manifest["marlin"] = info


def _load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _dense_all(data: dict, n: int, k: int) -> np.ndarray:
    codes = data["codes"]
    scales = data["scale_bytes"]
    gs = data["global_scale"]
    return np.stack(
        [dense_reference(codes[e], scales[e], float(gs[e])) for e in range(len(codes))]
    )


class Fixture:
    """Loaded fixture. Every tensor is a numpy array, ``reference`` included."""

    def __init__(self, directory: Path, manifest: dict, tensors: dict):
        self.directory = directory
        self.manifest = manifest
        self.tensors = tensors

    @property
    def shapes(self) -> tuple[str, ...]:
        return tuple(self.manifest["shapes"])

    def __getitem__(self, shape: str) -> dict:
        prefix = shape + "."
        return {
            key[len(prefix):]: value
            for key, value in self.tensors.items()
            if key.startswith(prefix)
        }

    def reference(self, shape: str) -> dict:
        """Dense fp32 [E, N, K] and expected fp32 [E, N] for this shape."""
        case = self.manifest["shapes"][shape]
        data = self[shape]
        return {
            "dense": _dense_all(data, case["n"], case["k"]),
            "expected": data["expected"],
        }


def load(directory: str | Path) -> Fixture:
    directory = Path(directory)
    with np.load(directory / "tensors.npz") as archive:
        tensors = {key: archive[key] for key in archive.files}
    return Fixture(directory, _load_manifest(directory), tensors)


def _iter_manifest_tensors(manifest: dict):
    for shape, case in manifest["shapes"].items():
        for key, entry in case["tensors"].items():
            yield f"{shape}.{key}", entry


def verify(directory: Path) -> None:
    """Rebuild every input from its stream and re-derive the references."""
    manifest = _load_manifest(directory)
    with np.load(directory / "tensors.npz") as archive:
        tensors = {key: archive[key] for key in archive.files}
    failures = 0

    for key, entry in _iter_manifest_tensors(manifest):
        got = tensors[key]
        source = "stored"
        if key.endswith((".codes", ".scale_bytes", ".global_scale", ".activation")):
            shape, field = key.split(".", 1)
            n, k = SHAPES[shape]
            source = "stream"
            fresh = {
                "codes": lambda: _codes(shape, manifest["experts"], n, k),
                "scale_bytes": lambda: _scale_bytes(shape, manifest["experts"], n, k),
                "global_scale": lambda: _global_scale(shape, manifest["experts"]),
                "activation": lambda: _activations(shape, manifest["route_rows"], k),
            }[field]()
            if not np.array_equal(fresh, got):
                failures += 1
                print(f"MISMATCH {key}: stored bytes differ from the stream")
                continue
        if _tensor_digest(got) != entry["sha256"]:
            failures += 1
            print(f"MISMATCH {key}: sha256 differs from the manifest")
            continue
        print(f"ok {key:28s} {str(got.shape):22s} {source}")

    for shape, case in manifest["shapes"].items():
        data = {
            key.split(".", 1)[1]: value
            for key, value in tensors.items()
            if key.startswith(shape + ".")
        }
        n, k = case["n"], case["k"]
        dense = _dense_all(data, n, k)
        if _tensor_digest(dense) != case["dense_sha256"]:
            failures += 1
            print(f"MISMATCH {shape}: dense reference sha256 differs")
        decoded = np.stack(
            [
                np.array(
                    oracle_qpn2.oracle_decode(
                        data["qpn2"][e].tolist(), n, k, float(data["global_scale"][e]),
                        fp16_scale=True,
                    ),
                    dtype=np.float32,
                )
                for e in range(len(dense))
            ]
        )
        if not np.array_equal(decoded, dense):
            failures += 1
            print(f"MISMATCH {shape}: QPN2 decode differs from the dense reference")
        expected = np.stack(
            [
                (dense[e].astype(np.float64) @ data["activation"][e].astype(np.float64)).astype(
                    np.float32
                )
                for e in range(len(dense))
            ]
        )
        if _tensor_digest(expected) != case["tensors"]["expected"]["sha256"]:
            failures += 1
            print(f"MISMATCH {shape}: expected outputs sha256 differs")
        pair_expected = (
            2.0 * (dense[0].astype(np.float64) @ data["activation"][0].astype(np.float64))
        ).astype(np.float32)
        if not np.array_equal(pair_expected, data["pair_expected"]):
            failures += 1
            print(f"MISMATCH {shape}: pair-case expected output differs")
        print(f"ok {shape}: QPN2 decode == dense reference, expected outputs match")

    if failures:
        raise SystemExit(f"{failures} fixture check(s) failed")
    print("fixture verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="write a fixture directory")
    build_p.add_argument("--out", required=True, type=Path)
    build_p.add_argument("--experts", type=int, default=EXPERTS)
    build_p.add_argument(
        "--with-marlin",
        action="store_true",
        help="run the shipping Marlin conversion once (needs a GPU and SM70_MARLIN_DIR)",
    )

    verify_p = sub.add_parser("verify", help="check an existing fixture")
    verify_p.add_argument("directory", type=Path)

    args = parser.parse_args()
    if args.command == "build":
        build(args.out, args.experts, args.with_marlin)
    else:
        verify(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
