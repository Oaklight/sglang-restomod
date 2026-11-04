"""The e5m2 to fp16 shift must equal the lookup table for every finite byte.

The verify and decode kernels convert e5m2 KV to FP16 before the MMA. The
decode kernel shifts the byte into the high half of a uint16 and reinterprets
it as FP16; the verify kernel used a 256-entry table until this change. E5M2
and FP16 share the sign and exponent bit positions, so the shift is exact.

The two disagree only for the six NaN encodings: torch's fp8 cast
canonicalises the NaN payload, and the shift preserves it. Both are NaN, and
KV data carries no NaNs, so the test compares bit patterns for the finite
values and asserts NaN on NaN.
"""

from __future__ import annotations

import pytest


def test_shift_reproduces_the_lut_for_all_finite_bytes():
    torch = pytest.importorskip("torch")
    raw = torch.arange(256, dtype=torch.uint8)

    lut = raw.view(torch.float8_e5m2).to(torch.float16)
    # torch has no uint16 left shift on CPU, so shift in int32 and narrow.
    shifted = (
        ((raw.to(torch.int32) << 8) & 0xFFFF)
        .to(torch.uint16)
        .view(torch.float16)
    )

    finite = ~torch.isnan(lut)
    assert torch.equal(
        lut.view(torch.uint16)[finite], shifted.view(torch.uint16)[finite]
    ), "the shift differs from the table on a finite e5m2 value"
    assert torch.isnan(shifted[~finite]).all(), "a NaN byte became finite"
