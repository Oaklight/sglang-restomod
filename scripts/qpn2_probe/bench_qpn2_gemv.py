"""QPN2 repack + direct-register GEMV probe (#224 P1).

Two gates, measured on one V100:

  * >= 400 GB/s on the gate_up weight stream at M=1 (45 us for 18.4 MB);
  * <= 1e-3 relative error against an FP32 dequantized reference.

The weights come from random E2M1 codes and E4M3 group scales packed with the
layout in docs/qpn2-layout.md. The reference is built from the same codes and
scales, not from the packed buffer, so a pack bug shows up as an error.

Run inside the image, on a free card:

    singularity exec --nv --bind /nfs:/nfs --bind /homes:/homes \
      <cuda-image>.sif \
      env CUDA_VISIBLE_DEVICES=<gpu> TORCH_CUDA_ARCH_LIST=7.0 \
        PATH=/usr/local/cuda/bin:<conda>/bin:/usr/bin:/bin \
      <conda>/bin/python scripts/qpn2_probe/bench_qpn2_gemv.py
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

# Flash-Next at TP4, as the server issues the call: 4 tokens x top_k 10 = 40
# route rows, one real row per 8-slot route tile.
SHAPES = {
    "gate_up": dict(n=320, k=2560, rows=40, tokens=4, experts=512),
    "down": dict(n=2560, k=160, rows=40, tokens=4, experts=512),
}


def qpn2_col_from_lane(lane: int) -> int:
    return ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if (lane & 16) else 0)


def e2m1_table() -> list[float]:
    return [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e4m3_table() -> list[float]:
    """E4M3 byte -> float, by hand: CPU float8 indexing is not implemented."""
    values = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            values.append(sign * mantissa * 2.0 ** -9)
        elif exponent == 15:
            # E4M3FN: NaN only at 0x7F / 0xFF. 0x78..0x7E are 256..448, which
            # the served checkpoint's down-projection scales reach (max 0x7E).
            values.append(
                float("nan") if mantissa == 7
                else sign * (1.0 + mantissa / 8.0) * 256.0
            )
        else:
            values.append(sign * 2.0 ** (exponent - 7) * (1.0 + mantissa / 8.0))
    return values


def pack_expert(codes: torch.Tensor, scale_bytes: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[n, k] E2M1 codes + [n, k/16] E4M3 bytes -> tiles*groups*288 uint8.

    Vectorized form of ``repack_qpn2.pack_qpn2``; that one is the CPU-verified
    reference for this layout.
    """
    groups, tiles = k // 16, n // 32
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    lanes = torch.arange(32, device=codes.device)
    cols = torch.tensor([qpn2_col_from_lane(int(lane)) for lane in lanes], device=codes.device)
    src = packed.view(tiles, 32, groups, 8).index_select(1, cols).to(torch.int16)
    lo, hi = src & 0xF, src >> 4
    codes_out = torch.zeros(tiles, groups, 32, 8, dtype=torch.int16, device=codes.device)
    for b in range(8):
        jb = 4 * ((b >> 2) & 1) + 2 * (b & 1)
        srcb = hi if (b >> 1) & 1 else lo
        codes_out[:, :, :, b] = (
            srcb[:, :, :, jb] | (srcb[:, :, :, jb + 1] << 4)).permute(0, 2, 1)
    scales = scale_bytes.view(tiles, 32, groups).index_select(1, cols).permute(0, 2, 1)
    out = torch.cat([codes_out.reshape(tiles, groups, 256).to(torch.uint8),
                     scales.to(torch.uint8)], dim=2)
    return out.reshape(-1)


def build_extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="qpn2_gemv_ext",
        cpp_sources="""
torch::Tensor decode_qpn2_gemv(torch::Tensor weights, torch::Tensor global_scale,
                               torch::Tensor activation, torch::Tensor expert_ids,
                               torch::Tensor token_ids, torch::Tensor out,
                               torch::Tensor workspace, int64_t split_k,
                               bool loads_only);
""",
        cuda_sources=(HERE / "decode_gemv_qpn2.cu").read_text(),
        functions=["decode_qpn2_gemv"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )


def build_case(shape: dict, device: torch.device, seed: int = 0) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, k, rows, tokens = shape["n"], shape["k"], shape["rows"], shape["tokens"]
    experts = shape["experts"]
    groups, tiles = k // 16, n // 32
    bytes_per_expert = tiles * groups * 288

    # One distinct expert per route row, as the existing Marlin probe does.
    expert_ids = torch.arange(rows, dtype=torch.int32, device=device)
    token_ids = torch.arange(rows, device=device, dtype=torch.int32) % tokens

    codes = torch.randint(0, 16, (rows, n, k), generator=gen, dtype=torch.int32)
    # Realistic group scales: exponents 6..8 (0.5..8) with random mantissas.
    scale_bytes = torch.randint(0, 8, (rows, n, k // 16), generator=gen, dtype=torch.int32)
    scale_bytes |= torch.randint(6, 9, (rows, n, k // 16), generator=gen, dtype=torch.int32) << 3
    gscale = (0.0123 * (1.0 + 0.1 * torch.rand(rows, generator=gen))).float()
    activation = torch.randn(tokens, k, generator=gen).to(torch.float16)

    weights = torch.zeros(experts * bytes_per_expert, dtype=torch.uint8, device=device)
    for row in range(rows):
        packed = pack_expert(codes[row].to(device), scale_bytes[row].to(device), n, k)
        weights[row * bytes_per_expert:(row + 1) * bytes_per_expert] = packed
    weights = weights.view(experts, bytes_per_expert)

    return dict(
        n=n, k=k, rows=rows, tokens=tokens, tiles=tiles, groups=groups,
        bytes_per_expert=bytes_per_expert, expert_ids=expert_ids,
        token_ids=token_ids, weights=weights,
        gscale=gscale.to(device), activation=activation.to(device),
        codes=codes, scale_bytes=scale_bytes,
        out=torch.empty(rows, n, dtype=torch.float16, device=device),
        workspace=torch.empty(32 * rows * n, dtype=torch.float32, device=device),
    )


def reference(case: dict, fp16_scale: bool = False) -> torch.Tensor:
    """Dense reference from the source codes and scales.

    ``fp16_scale=False`` is the exact FP32 product (the mathematical value of
    the checkpoint). ``fp16_scale=True`` follows the spec's kernel path: round
    ``e4m3 * global_scale`` once to FP16, then round ``code * scale`` once.
    """
    e2m1 = torch.tensor(e2m1_table(), dtype=torch.float32, device=case["weights"].device)
    e4m3 = torch.tensor(e4m3_table(), dtype=torch.float32, device=case["weights"].device)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.empty(case["rows"], case["n"], dtype=torch.float32, device=e2m1.device)
    for row in range(case["rows"]):
        device = e2m1.device
        w = e2m1[case["codes"][row].to(device).long()]
        s = e4m3[case["scale_bytes"][row].to(device).long()]
        s = s * case["gscale"][row]
        if fp16_scale:
            s = s.half().float()
        s = s.repeat_interleave(16, dim=1)
        w = (w * s).half().float() if fp16_scale else w * s
        x = case["activation"][case["token_ids"][row].item()].float()
        out[row] = torch.mv(w, x)
    torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=tuple(SHAPES) + ("both",), default="both")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--split-k", type=int, default=4)
    parser.add_argument("--loads-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    device = torch.device("cuda")
    ext = build_extension()
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")

    names = list(SHAPES) if args.shape == "both" else [args.shape]
    for name in names:
        shape = SHAPES[name]
        case = build_case(shape, device)
        want = reference(case)
        want16 = reference(case, fp16_scale=True)

        def run(loads_only: bool):
            return ext.decode_qpn2_gemv(
                case["weights"], case["gscale"], case["activation"],
                case["expert_ids"], case["token_ids"], case["out"],
                case["workspace"], args.split_k, loads_only)

        got = run(args.loads_only).float()
        torch.cuda.synchronize()
        scale = want.abs().max().item()
        diff = (got - want).abs().max().item()
        diff16 = (got - want16).abs().max().item()
        weight_bytes = case["rows"] * case["bytes_per_expert"]

        arms = (("compute", lambda: run(False)), ("loads", lambda: run(True)))
        if args.loads_only:
            arms = arms[1:]
        for _, fn in arms:
            for _ in range(args.warmup):
                fn()
        torch.cuda.synchronize()

        samples = {label: [] for label, _ in arms}
        for _ in range(args.rounds):
            for label, fn in arms:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    fn()
                end.record()
                torch.cuda.synchronize()
                samples[label].append(start.elapsed_time(end) / args.iters * 1e3)

        print(f"\n{name}: N={case['n']} K={case['k']} rows={case['rows']} "
              f"({weight_bytes / 1e6:.1f} MB of weights, split_k={args.split_k})")
        print(f"  error   vs exact fp32  max|d|/max|ref| = {diff / scale:.3e}")
        print(f"          vs fp16-RNE    max|d|/max|ref| = {diff16 / scale:.3e}")
        for label, _ in arms:
            values = samples[label]
            us = statistics.median(values)
            print(f"  {label:<8} {us:7.1f} us  (min {min(values):.1f} max {max(values):.1f})  "
                  f"{weight_bytes / (us * 1e-6) / 1e9:5.0f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
