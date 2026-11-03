"""CPU test for the GPTQ-to-marlin_v100 argument mapping.

The extension's ``_C::marlin_gemm`` has a bias slot and an activation-scale
slot that the engine's ``gptq_marlin_gemm`` does not carry, so the call is
rebuilt positionally. Getting a slot wrong shifts every later argument and
produces plausible-looking garbage, which is how the scale-order bug was
found, so the mapping is pinned here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_restomod.quant.sm70_marlin_v100 import gemm_args  # noqa: E402


class _Type:
    id = 1049600


def test_empty_slots_are_none():
    t = torch.zeros(1)
    args = gemm_args(
        t, None, t, t, None, None, None, None, t, _Type(), 6, 5120, 4096, True,
        False, False, False,
    )
    assert args[3] is None  # bias
    assert args[5] is None  # activation scales


def test_order_matches_the_extension_signature():
    a, c, w, s, gs, z, g, p, ws = (torch.zeros(1) for _ in range(9))
    args = gemm_args(
        a, c, w, s, gs, z, g, p, ws, _Type(), 6, 5120, 4096, True, False, True, False
    )
    assert args[:4] == (a, c, w, None)
    assert args[4:12] == (s, None, gs, z, g, p, ws, 1049600)
    assert args[12:16] == (6, 5120, 4096, True)
    assert args[16:19] == (False, True, False)
