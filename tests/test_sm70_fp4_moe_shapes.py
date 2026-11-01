"""Shape-gate tests for the SM70 NVFP4 expert repack (CPU).

``check_expert_shape`` runs before the extension loads, so the gate is
reachable without a GPU or a ``SM70_MARLIN_DIR``. Both sides of the boundary
matter: the shipped TP4 profile must pass, and the eight-way shard the
reference pads must be rejected by name.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _gate():
    from sglang_restomod.quant.sm70_fp4_moe.method import check_expert_shape

    return check_expert_shape


def test_tp4_profile_is_accepted():
    gate = _gate()
    # moe_intermediate_size=640 over TP=4 -> inter=160 per rank.
    gate(size_n=320, size_k=2560)  # w13: 2*inter, hidden
    gate(size_n=2560, size_k=160)  # w2: hidden, inter


def test_eight_way_shard_is_rejected():
    gate = _gate()
    # inter=80 per rank: w13's N is 160, and 160 % 64 == 32.
    with pytest.raises(RuntimeError, match=r"N=160"):
        gate(size_n=160, size_k=2560)


def test_k_not_divisible_by_16_is_rejected():
    gate = _gate()
    with pytest.raises(RuntimeError, match=r"K=8\b"):
        gate(size_n=320, size_k=8)
