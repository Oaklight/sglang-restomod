"""The attest record must distinguish "registered and inactive" from both
"loaded" and "failed".

On 2026-11-15 three components took the inactive path at once (the SM70
block-FP8 library was not resolvable), the server served on bf16 dense linears
under the same component list, and every gate then available reported success.
These tests pin the state that was missing.
"""

from __future__ import annotations

import pytest


def _attest():
    pytest.importorskip("torch")
    from sglang_restomod import attest

    attest.reset()
    return attest


def test_inactive_is_its_own_state():
    attest = _attest()

    attest.record("stub", ok=True)
    attest.inactive("stub", "library missing")

    snapshot = attest.snapshot()
    assert snapshot["inactive"] == ["stub"]
    # Loaded, because it registered; not failed, because it did not raise.
    assert "stub" in snapshot["loaded"]
    assert snapshot["failed"] == []
    assert snapshot["components"]["stub"]["detail"] == "registered, inactive: library missing"


def test_inactive_never_raises_on_a_bad_reason():
    attest = _attest()

    class Hostile:
        def __str__(self):
            raise RuntimeError("no")

    attest.inactive("stub", Hostile())  # must not propagate

    snapshot = attest.snapshot()
    assert "stub" in snapshot["inactive"]
    assert snapshot["components"]["stub"]["detail"].endswith("<unprintable reason>")


def test_a_component_that_skips_is_recorded_by_its_own_install(monkeypatch):
    """fp4_moe without a loadable extension registers but does nothing.

    The environment is forced so the outcome does not depend on whether the
    machine running the suite has an SM70 card.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sglang.srt.layers.quantization.modelopt_quant")
    from sglang_restomod import attest
    from sglang_restomod.quant.sm70_fp4_moe import method

    attest.reset()
    monkeypatch.setattr(method, "available", lambda: False)
    monkeypatch.setattr(method, "_INSTALLED", False)
    monkeypatch.delenv("SM70_MARLIN_DIR", raising=False)
    monkeypatch.setenv("SGLANG_SM70_FP4_MOE", "auto")

    method.install()

    snapshot = attest.snapshot()
    assert "fp4_moe" in snapshot["inactive"]
    assert "fp4_moe" not in snapshot["failed"]
    assert snapshot["components"]["fp4_moe"]["detail"].startswith("registered, inactive:")


def test_record_after_inactive_keeps_the_reason():
    """The plugin records ``ok`` after ``register()`` returns.

    A component that called ``attest.inactive`` from inside ``register()`` had
    its reason overwritten by that later ``record`` call, so ``inactive`` named
    the component and said nothing about why.
    """
    attest = _attest()

    attest.inactive("stub", "no flashinfer binding")
    attest.record("stub", ok=True)

    snapshot = attest.snapshot()
    assert snapshot["inactive"] == ["stub"]
    assert snapshot["components"]["stub"]["detail"] == (
        "registered, inactive: no flashinfer binding"
    )


def test_record_with_its_own_detail_replaces_the_old_one():
    attest = _attest()

    attest.inactive("stub", "no flashinfer binding")
    attest.record("stub", ok=False, detail="RuntimeError: boom")

    snapshot = attest.snapshot()
    assert snapshot["components"]["stub"]["detail"] == "RuntimeError: boom"
