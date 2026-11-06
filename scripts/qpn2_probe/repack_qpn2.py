"""QPN2 NVFP4 repack and its CPU round-trip check.

The layout is defined in ``docs/qpn2-layout.md``; this module is the reference
implementation the probe and the CUDA kernel are checked against. Pure torch,
no CUDA: it runs on the laptop.
"""

from __future__ import annotations

import torch

# E2M1 code (bit 3 = sign) -> value.
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def qpn2_col_from_lane(lane: int) -> int:
    return ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if (lane & 16) else 0)


def qpn2_logical_k(physical_k: int) -> int:
    local = physical_k & 7
    return (physical_k & 8) + ((local & 3) << 1) + (local >> 2)


def qpn2_lane_from_col(col: int) -> int:
    return ((col & 24) >> 1) | (col & 3) | ((col & 4) << 2)


def e4m3_to_float(byte: int) -> float:
    """Exact E4M3 byte -> Python float."""
    t = torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn)
    return float(t.float().item())


def pack_qpn2(weight_packed: torch.Tensor, scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """ModelOpt packed ``[N, K/2]`` uint8 + E4M3 ``[N, K/16]`` -> QPN2 bytes.

    The output is ``[tiles*groups*288]`` uint8; the per-expert global scale is
    not baked in.
    """
    if weight_packed.shape != (n, k // 2):
        raise ValueError(f"weight_packed must be [{n}, {k // 2}], got {tuple(weight_packed.shape)}")
    if scales.shape != (n, k // 16):
        raise ValueError(f"scales must be [{n}, {k // 16}], got {tuple(scales.shape)}")
    if n % 32 or k % 16:
        raise ValueError(f"QPN2 needs N%32==0 and K%16==0, got N={n} K={k}")
    groups, tiles = k // 16, n // 32
    # CPU float8 has no index kernels; the bytes are the same either way.
    scales_u8 = scales.view(torch.uint8)
    lanes = torch.arange(32)
    physical = torch.arange(16)
    logical = torch.tensor([qpn2_logical_k(int(p)) for p in physical])
    cols_in_tile = torch.tensor([qpn2_col_from_lane(int(l)) for l in lanes])

    out = torch.zeros(tiles * groups * 288, dtype=torch.uint8)
    for t in range(tiles):
        cols = t * 32 + cols_in_tile
        for g in range(groups):
            base = (t * groups + g) * 288
            kk = g * 16 + logical  # [16] logical K in physical order
            src = weight_packed[cols][:, kk // 2]  # [32 lanes, 16]
            shift = (kk & 1) * 4
            codes = (src >> shift) & 0xF  # [32, 16], physical order
            lo = codes[:, 0::2]  # physical 0,2,4,...
            hi = codes[:, 1::2]  # physical 1,3,5,...
            out[base:base + 256] = (lo | (hi << 4)).to(torch.uint8).reshape(-1)
            out[base + 256:base + 288] = scales_u8[cols, g]
    return out


def dequant_qpn2(buf: torch.Tensor, n: int, k: int, global_scale: float) -> torch.Tensor:
    """QPN2 bytes -> FP32 dense ``[N, K]``, the reference for the probe."""
    groups, tiles = k // 16, n // 32
    out = torch.zeros(n, k, dtype=torch.float32)
    for t in range(tiles):
        cols = t * 32 + torch.tensor([qpn2_col_from_lane(l) for l in range(32)])
        for g in range(groups):
            base = (t * groups + g) * 288
            codes = buf[base:base + 256].reshape(32, 8)
            scales = buf[base + 256:base + 288]
            for lane in range(32):
                scale = e4m3_to_float(int(scales[lane])) * global_scale
                col = int(cols[lane])
                for b in range(8):
                    byte = int(codes[lane, b])
                    for half, code in ((0, byte & 0xF), (1, byte >> 4)):
                        kk = g * 16 + qpn2_logical_k(2 * b + half)
                        out[col, kk] = float(E2M1[code]) * scale
    return out


def dequant_source(weight_packed: torch.Tensor, scales: torch.Tensor,
                   n: int, k: int, global_scale: float) -> torch.Tensor:
    """Direct decode of the source, independent of the QPN2 permutation."""
    scales_u8 = scales.view(torch.uint8)
    out = torch.zeros(n, k, dtype=torch.float32)
    for col in range(n):
        for kk in range(k):
            byte = int(weight_packed[col, kk // 2])
            code = (byte >> ((kk & 1) * 4)) & 0xF
            scale = e4m3_to_float(int(scales_u8[col, kk // 16]))
            out[col, kk] = float(E2M1[code]) * scale * global_scale
    return out


def _round_trip(n: int, k: int, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.int32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale_bytes = torch.randint(0, 256, (n, k // 16), generator=g, dtype=torch.int32)
    # Keep the E4M3 values finite: clear NaN/Inf encodings (exp all ones).
    scale_bytes = torch.where((scale_bytes & 0x7C) == 0x7C, scale_bytes & ~0x7C, scale_bytes)
    scales = scale_bytes.to(torch.uint8).view(torch.float8_e4m3fn)
    gs = 0.0123

    buf = pack_qpn2(packed, scales, n, k)
    got = dequant_qpn2(buf, n, k, gs)
    want = dequant_source(packed, scales, n, k, gs)
    diff = (got - want).abs().max().item()
    print(f"N={n} K={k}: bytes={buf.numel()} max|qpn2-source|={diff:.3e}")
    assert diff == 0.0, "QPN2 round trip is not bit-exact against the source decode"


if __name__ == "__main__":
    _round_trip(64, 32)
    _round_trip(320, 64)
    _round_trip(64, 160)
    print("QPN2 repack round trip OK")
