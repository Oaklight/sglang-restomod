#!/usr/bin/env python3
"""Independent QPN2 oracle, transcribed from ``docs/qpn2-layout.md`` only.

The oracle functions use only the Python standard library and do not import
``repack_qpn2``. The comparison section imports 9b's module explicitly, only to
obtain its bytes and dense decode for the diff. Nothing in 9b's code is
modified.

Run: ``python oracle_qpn2.py`` (writes the report to stdout).
"""

from __future__ import annotations

import os
import random
import struct
import sys

# Spec section 3: E2M1 code (bit 3 = sign) -> value.
E2M1 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


# --- Section 1: lane <-> column -------------------------------------------

def col_from_lane(lane: int) -> int:
    return ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if (lane & 16) else 0)


def lane_from_col(col: int) -> int:
    return ((col & 24) >> 1) | (col & 3) | ((col & 4) << 2)


# --- Section 2: physical -> logical K -------------------------------------

def logical_k(physical: int) -> int:
    local = physical & 7
    return (physical & 8) + ((local & 3) << 1) + (local >> 2)


# --- Section 3: E4M3 byte -> scale, with the spec's one fp16 rounding ------

def e4m3_raw(byte: int) -> float:
    """The spec's bit twiddle: shift the 7-bit magnitude into a half, *256."""
    bits = (((byte & 0x80) << 8) | ((byte & 0x7f) << 7)) & 0xFFFF
    return struct.unpack("<e", struct.pack("<H", bits))[0] * 256.0


def fp16_rne(x: float) -> float:
    return struct.unpack("<e", struct.pack("<e", x))[0]


def spec_scale(byte: int, global_scale: float) -> float:
    """Spec section 3: round raw * global_scale to fp16 exactly once."""
    return fp16_rne(e4m3_raw(byte) * global_scale)


def reference_scale(byte: int, global_scale: float) -> float:
    """9b's reference path: float32 multiply, no fp16 rounding."""
    return e4m3_raw(byte) * global_scale


# --- Section 4/5: repack and decode ---------------------------------------

def tiles_for(n: int) -> int:
    # Section 5 says pad a partial last tile; tiles is ceil(N/32).
    return (n + 31) // 32


def oracle_pack(weight, scales, n: int, k: int):
    """Spec section 4, with the section 5 padded-column rule.

    ``weight``: n rows of ``k//2`` packed bytes (low nibble = even K).
    ``scales``:  n rows of ``k//16`` E4M3 bytes.
    Returns a ``bytearray`` of ``ceil(n/32) * (k//16) * 288`` bytes.
    """
    if len(weight) != n or len(weight[0]) != k // 2:
        raise ValueError("weight shape mismatch")
    if len(scales) != n or len(scales[0]) != k // 16:
        raise ValueError("scales shape mismatch")
    groups, tiles = k // 16, tiles_for(n)
    out = bytearray(tiles * groups * 288)
    for t in range(tiles):
        for g in range(groups):
            base = (t * groups + g) * 288
            for lane in range(32):
                col = t * 32 + col_from_lane(lane)
                if col >= n:
                    # Padded column: the allocation's zeros are the contract.
                    continue
                for b in range(8):
                    p0, p1 = 2 * b, 2 * b + 1
                    k0, k1 = g * 16 + logical_k(p0), g * 16 + logical_k(p1)
                    c0 = (weight[col][k0 // 2] >> ((k0 & 1) * 4)) & 0xF
                    c1 = (weight[col][k1 // 2] >> ((k1 & 1) * 4)) & 0xF
                    out[base + lane * 8 + b] = c0 | (c1 << 4)
                out[base + 256 + lane] = scales[col][g]
    return out


def oracle_decode(buf, n: int, k: int, global_scale: float, fp16_scale: bool = True):
    """Spec sections 1-3: QPN2 bytes -> dense [n][k] floats."""
    groups, tiles = k // 16, tiles_for(n)
    out = [[0.0] * k for _ in range(n)]
    for t in range(tiles):
        for g in range(groups):
            base = (t * groups + g) * 288
            for lane in range(32):
                col = t * 32 + col_from_lane(lane)
                if col >= n:
                    continue
                sb = buf[base + 256 + lane]
                scale = spec_scale(sb, global_scale) if fp16_scale else reference_scale(sb, global_scale)
                for b in range(8):
                    byte = buf[base + lane * 8 + b]
                    for half, code in ((0, byte & 0xF), (1, byte >> 4)):
                        p = 2 * b + half
                        kk = g * 16 + logical_k(p)
                        out[col][kk] = E2M1[code] * scale
    return out


def source_decode(weight, scales, n: int, k: int, global_scale: float, fp16_scale: bool = True):
    """Direct decode of the ModelOpt source, independent of every QPN2 rule."""
    out = [[0.0] * k for _ in range(n)]
    for col in range(n):
        for kk in range(k):
            byte = weight[col][kk // 2]
            code = (byte >> ((kk & 1) * 4)) & 0xF
            sb = scales[col][kk // 16]
            scale = spec_scale(sb, global_scale) if fp16_scale else reference_scale(sb, global_scale)
            out[col][kk] = E2M1[code] * scale
    return out


# --- Test helpers ---------------------------------------------------------

def make_case(n: int, k: int, seed: int):
    rng = random.Random(seed)
    codes = [[rng.randrange(16) for _ in range(k)] for _ in range(n)]
    weight = [
        [codes[c][2 * i] | (codes[c][2 * i + 1] << 4) for i in range(k // 2)]
        for c in range(n)
    ]
    scales = []
    for c in range(n):
        row = []
        for _ in range(k // 16):
            b = rng.randrange(256)
            if (b & 0x7F) == 0x7F:  # E4M3 NaN encoding
                b &= 0x80
            row.append(b)
        scales.append(row)
    return weight, scales


def max_abs(a, b) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def first_byte_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return None


# --- Tests ----------------------------------------------------------------

def test_arithmetic():
    print("== 1. 288-byte group, lane/column map, K order, section-4 addresses")
    assert 32 * 16 * 4 // 8 + 32 == 288
    assert 288 / 512 == 0.5625
    lane_cols = [col_from_lane(l) for l in range(32)]
    assert lane_cols == [
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31,
    ]
    assert all(lane_from_col(col_from_lane(l)) == l for l in range(32))
    assert all(col_from_lane(lane_from_col(c)) == c for c in range(32))
    assert [logical_k(p) for p in range(16)] == [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    # Section 4 decomposes the *code-byte index* i, not a buffer offset. The
    # buffer address adds the 32-byte scale region: group*288 + lane*8 + b.
    t, g, lane, b = 1, 2, 3, 5
    i = (((t * 4 + g) * 32) + lane) * 8 + b  # groups=4 for this decomposition
    d_b, outer = i & 7, i >> 3
    d_lane, outer = outer & 31, outer >> 5
    d_g, d_t = outer % 4, outer // 4
    assert (d_t, d_g, d_lane, d_b) == (t, g, lane, b)
    assert ((t * 4 + g) * 288 + lane * 8 + b) != i
    print("   288 B group, 32 lane<->column pairs, K bijection, and the")
    print("   code-index vs buffer-address distinction all hold.")


def test_supported(n: int, k: int):
    gs = 0.0123
    weight, scales = make_case(n, k, seed=n * 1000 + k)
    buf = oracle_pack(weight, scales, n, k)
    src = source_decode(weight, scales, n, k, gs, fp16_scale=False)
    dec = oracle_decode(buf, n, k, gs, fp16_scale=False)
    print(f"== 2. supported N={n} K={k}: oracle round trip max|d|={max_abs(dec, src):.3e}")
    assert max_abs(dec, src) == 0.0

    # 9b's repack, for the byte diff and an independent decode of its bytes.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import repack_qpn2  # noqa: E402  (comparison target only)

    w_t = torch.tensor(weight, dtype=torch.uint8)
    s_t = torch.tensor(scales, dtype=torch.uint8).view(torch.float8_e4m3fn)
    nineb = repack_qpn2.pack_qpn2(w_t, s_t, n, k)
    nineb_bytes = bytes(nineb.tolist())
    mismatch = first_byte_mismatch(buf, nineb_bytes)
    print(f"   oracle bytes vs 9b bytes: {'identical' if mismatch is None else 'MISMATCH at %r' % (mismatch,)}")
    dec_9b = oracle_decode(nineb_bytes, n, k, gs, fp16_scale=False)
    print(f"   oracle decode of 9b bytes vs source: max|d|={max_abs(dec_9b, src):.3e}")
    own = repack_qpn2.dequant_qpn2(nineb, n, k, gs)
    print(f"   9b decode vs oracle decode of the same bytes: max|d|={max_abs(own.tolist(), dec_9b):.3e}")


def test_fp16_scale():
    print("== 3. fp16-RNE scale vs the reference float32 multiply")
    best = None
    gs_candidates = [0.0123, 0.1, 0.333333, 1.7, 3.14159, 0.0001, 123.4, 7.77]
    for b in range(256):
        if (b & 0x7F) == 0x7F:
            continue
        for gs in gs_candidates:
            a, c = spec_scale(b, gs), reference_scale(b, gs)
            if a != c and (best is None or abs(a - c) > abs(best[2] - best[3])):
                best = (b, gs, a, c)
    found = best
    assert found is not None, "no distinguishing case found"
    b, gs, a, c = found
    print(f"   byte=0x{b:02x} gs={gs}: spec={a!r} float32={c!r} delta={a - c:.3e}")
    # One K16 group with that scale byte, to show the decode difference.
    n, k = 32, 16
    weight, scales = make_case(n, k, seed=7)
    scales = [[b] * (k // 16) for _ in range(n)]
    buf = oracle_pack(weight, scales, n, k)
    spec = oracle_decode(buf, n, k, gs, fp16_scale=True)
    ref = oracle_decode(buf, n, k, gs, fp16_scale=False)
    print(f"   full decode: spec-fp16 vs reference-fp32 max|d|={max_abs(spec, ref):.3e}")


def test_padded():
    print("== 4. padded N (40) with explicit zero lanes and scale")
    n, k = 40, 32
    gs = 0.0123
    weight, scales = make_case(n, k, seed=11)
    buf = oracle_pack(weight, scales, n, k)
    tiles, groups = tiles_for(n), k // 16
    assert len(buf) == tiles * groups * 288
    padded_zero = True
    for t in range(tiles):
        for g in range(groups):
            base = (t * groups + g) * 288
            for lane in range(32):
                col = t * 32 + col_from_lane(lane)
                if col < n:
                    continue
                if any(buf[base + lane * 8 + b] for b in range(8)):
                    padded_zero = False
                if buf[base + 256 + lane] != 0:
                    padded_zero = False
    src = source_decode(weight, scales, n, k, gs, fp16_scale=False)
    dec = oracle_decode(buf, n, k, gs, fp16_scale=False)
    print(f"   tiles={tiles} bytes={len(buf)} padded lanes zero: {padded_zero}")
    print(f"   defined columns match source: {max_abs(dec[:n], src) == 0.0}")
    # 9b's repack rejects N%32; record it rather than patching it.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import repack_qpn2  # noqa: E402
    try:
        repack_qpn2.pack_qpn2(
            torch.tensor(weight, dtype=torch.uint8),
            torch.tensor(scales, dtype=torch.uint8).view(torch.float8_e4m3fn),
            n, k,
        )
        print("   9b repack: accepted N=40 (unexpected)")
    except ValueError as exc:
        print(f"   9b repack: rejected N=40 -> {exc}")


def _expert_pack(weight, scales, n, k, experts):
    """Stack e identical-shape experts contiguously (section 4.1)."""
    one = oracle_pack(weight, scales, n, k)
    return b"".join(one for _ in range(experts))


def test_tp4():
    print("== 5. TP4 w13 (tiles split) and w2 (groups split)")
    gs = 0.0123
    # w13: full N=1280, K=2560; per rank N=320, K=2560.
    n_full, k13, tp = 1280, 2560, 4
    w13, s13 = make_case(n_full, k13, seed=21)
    full13 = oracle_pack(w13, s13, n_full, k13)
    g13, t13 = k13 // 16, n_full // 32
    ok13 = True
    for r in range(tp):
        n_r = n_full // tp
        rank_w = w13[r * n_r:(r + 1) * n_r]
        rank_s = s13[r * n_r:(r + 1) * n_r]
        rank_pack = oracle_pack(rank_w, rank_s, n_r, k13)
        # Full layout is tile-major inside each expert, so the rank's tiles are
        # one contiguous byte range per expert.
        lo = (r * (n_r // 32) * g13) * 288
        hi = lo + (n_r // 32) * g13 * 288
        ok13 &= full13[lo:hi] == rank_pack
    print(f"   w13 N=1280->320: per-rank tiles contiguous and byte-identical: {ok13}")

    # w2: full N=2560, K=640; per rank N=2560, K=160 (groups split).
    n2, k2 = 2560, 640
    w2, s2 = make_case(n2, k2, seed=22)
    full2 = oracle_pack(w2, s2, n2, k2)
    g2, t2 = k2 // 16, n2 // 32
    ok2 = True
    for r in range(tp):
        k_r = k2 // tp
        rank_w = [row[r * (k_r // 2):(r + 1) * (k_r // 2)] for row in w2]
        rank_s = [row[r * (k_r // 16):(r + 1) * (k_r // 16)] for row in s2]
        rank_pack = oracle_pack(rank_w, rank_s, n2, k_r)
        # Groups split is NOT contiguous: each tile contributes a 10-group run.
        for t in range(t2):
            full_seg = full2[(t * g2 + r * (k_r // 16)) * 288:(t * g2 + r * (k_r // 16) + k_r // 16) * 288]
            rank_seg = rank_pack[(t * (k_r // 16)) * 288:(t + 1) * (k_r // 16) * 288]
            ok2 &= full_seg == rank_seg
    print(f"   w2 K=640->160: per-rank groups match, strided per tile: {ok2}")
    print("   repack-after sharding gives the rank's own stride directly;")
    print("   repack-before needs a per-tile copy for w2 and a contiguous one for w13.")

    # Section 4.1 expert stride: two experts concatenate at tiles*groups*288.
    one = oracle_pack(w13[:320], s13[:320], 320, k13)
    two = _expert_pack(w13[:320], s13[:320], 320, k13, 2)
    print(f"   expert stride {len(one)} bytes, 2-expert buffer matches: {two == one + one}")


def main():
    test_arithmetic()
    for n, k in ((64, 32), (320, 64), (64, 160)):
        test_supported(n, k)
    test_fp16_scale()
    test_padded()
    test_tp4()
    print("oracle checks complete")


if __name__ == "__main__":
    main()
