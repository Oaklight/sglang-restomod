"""Bit-exactness check for the fused QPN2 -> Marlin repack.

Reference: ``qpn2.unpack_expert`` (exact inverse of pack_expert) followed by the
production ``_repack_experts``. The fused kernel must match it word for word on
both TP4 shapes. Kept permanently: it is what makes a future rewrite of the
kernel safe.
"""

from __future__ import annotations

import sys

import torch
from torch.utils.cpp_extension import load

from sglang_restomod.quant.sm70_fp4_moe import qpn2
from sglang_restomod.quant.sm70_fp4_moe.method import _repack_experts

HERE = "sglang_restomod/quant/sm70_fp4_moe/csrc"

SHAPES = ((320, 2560), (2560, 160))


def main() -> int:
    ext = load(
        name="repack_qpn2_marlin",
        sources=[f"{HERE}/repack_qpn2_marlin.cu"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    bad = 0
    for n, k in SHAPES:
        experts = 8
        torch.manual_seed(11)
        bank = torch.randint(
            0, 256, (experts, (n // 32) * (k // 16) * 288),
            dtype=torch.uint8, device="cuda",
        )
        got = ext.repack_qpn2_marlin(bank.reshape(-1), n, k)
        codes, scales = qpn2.unpack_expert(bank, n, k)
        want = _repack_experts(codes.contiguous())
        ok = torch.equal(got, want)
        print(f"N={n} K={k} E={experts}: fused == reference: {ok} "
              f"({got.shape[0]}x{got.shape[1]}x{got.shape[2]})")
        if not ok:
            diff = (got != want).nonzero()
            print("  first mismatch:", diff[0].tolist() if len(diff) else "none")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
