"""The e5m2 to fp16 shift must equal the lookup table for every byte.

The verify and decode kernels convert e5m2 KV to FP16 before the MMA. The
decode kernel shifts the byte into the high half of a uint16 and reinterprets
it as FP16; the verify kernel used a 256-entry table until this change. E5M2
and FP16 share the sign and exponent bit positions, so the shift is exact.
This test compares the two conversions as bit patterns, which is the only
comparison that survives the NaN encodings.
"""

from __future__ import annotations

import pytest


def test_shift_reproduces_the_lut_for_all_256_bytes():
    torch = pytest.importorskip("torch")
    raw = torch.arange(256, dtype=torch.uint8)
    lut = raw.view(torch.float8_e5m2).to(torch.float16).view(torch.uint16)
    shifted = (raw.to(torch.uint16) << 8).view(torch.float16).view(torch.uint16)
    bad = (lut != shifted).nonzero().flatten().tolist()
    assert bad == [], f"shift differs from the table at byte values {bad}"
