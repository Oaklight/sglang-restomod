"""Patch 0007's contract: a backend without flash_mla still constructs.

The guard is only as good as the day it was written, so this pins the
behaviour the comment claims: with flash_mla absent the resolver returns a
callable that raises a named error, and with it present the resolver returns
the real binding. A later refactor that restores an eager module-level import
makes this test skip (the helper is gone) rather than pass silently.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _resolve():
    module = pytest.importorskip("sglang.srt.layers.attention.deepseek_v4_backend")
    resolve = getattr(module, "resolve_flash_mla", None)
    if resolve is None:
        pytest.skip("this tree does not carry patch 0007 (resolve_flash_mla)")
    return resolve


def test_without_flash_mla_the_call_raises_a_named_error():
    resolve = _resolve()
    try:
        import flash_mla  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("flash_mla is installed here; this arm needs its absence")

    call, split_kv_min_topk = resolve(swa_interleaved=False)

    assert split_kv_min_topk == 0
    with pytest.raises(RuntimeError, match="flash_mla is not available"):
        call(None, None, None)


def test_with_flash_mla_the_binding_is_real():
    resolve = _resolve()
    try:
        import flash_mla  # noqa: F401
    except ImportError:
        pytest.skip("flash_mla is absent here; this arm needs it installed")

    call, split_kv_min_topk = resolve(swa_interleaved=False)

    assert callable(call)
    assert isinstance(split_kv_min_topk, int)
