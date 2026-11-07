"""CPU tests for the v0.5.23 routing hints on the SM70 expert GEMM.

v0.5.23's ``moe_wna16_marlin_gemm`` takes three hints the SM70 extension does
not have. Two of them only pick the pin's sorted specialization and are
ignored; ``zero_inactive_output`` is a promise to the caller that the rows the
routing did not fill are zero, and the SM70 kernel does not write them. The
check here is what this module hands the extension: a zeroed buffer, or the
caller's own buffer zeroed in place. The extension is faked, so this runs on a
CPU with no ``SM70_MARLIN_DIR``.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

# The pin passes the sgl_kernel ScalarType; the SM70 call site reads only its
# id, so the test passes any object with one.
_BQ_TYPE = types.SimpleNamespace(id=0, name="float8_e4m3fn")


class _FakeOp:
    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return args[1]


class _FakeExtension:
    def __init__(self):
        self.gemm = _FakeOp()


def _call(monkeypatch, *, c_or_none, zero_inactive_output, **hints):
    from sglang_restomod.quant.sm70_fp4_moe import method

    extension = _FakeExtension()
    monkeypatch.setattr(method, "ensure_loaded", lambda: extension)

    size_m, size_n, size_k, top_k = 3, 64, 64, 2
    return _run(
        method,
        extension,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        top_k=top_k,
        c_or_none=c_or_none,
        zero_inactive_output=zero_inactive_output,
        hints=hints,
    )


def _run(
    method, extension, *, size_m, size_n, size_k, top_k, c_or_none, zero_inactive_output, hints
):
    method.moe_wna16_marlin_gemm(
        torch.zeros(2, 32, dtype=torch.half),  # a
        c_or_none,
        torch.zeros(1, 1, dtype=torch.int32),  # b_q_weight
        # A bias makes the call skip the decode-GEMV shortcut, which would
        # otherwise intercept a small batch and never reach the extension.
        torch.zeros(1, dtype=torch.half),  # b_bias_or_none
        torch.zeros(1, dtype=torch.half),  # b_scales
        None,  # global_scale_or_none
        None,  # b_zeros_or_none
        None,  # g_idx_or_none
        None,  # perm_or_none
        torch.zeros(1, dtype=torch.int32),  # workspace
        torch.zeros(8, dtype=torch.int32),  # sorted_token_ids
        torch.zeros(2, dtype=torch.int32),  # expert_ids
        torch.zeros(1, dtype=torch.int32),  # num_tokens_post_padded
        torch.zeros(8, dtype=torch.half),  # topk_weights
        64,  # moe_block_size
        top_k,
        True,  # mul_topk_weights
        False,  # is_ep
        _BQ_TYPE,  # b_q_type; the extension reads only .id
        size_m,
        size_n,
        size_k,
        zero_inactive_output=zero_inactive_output,
        **hints,
    )
    return extension.gemm.args


def test_the_callers_buffer_is_zeroed_in_place(monkeypatch):
    c = torch.full((6, 64), 7.0, dtype=torch.half)
    args = _call(monkeypatch, c_or_none=c, zero_inactive_output=True)

    assert args[1] is c, "the caller's buffer must be reused, not replaced"
    assert torch.all(c == 0), "inactive rows the kernel does not write must be zero"


def test_a_missing_buffer_is_allocated_zeroed(monkeypatch):
    args = _call(monkeypatch, c_or_none=None, zero_inactive_output=True)

    out = args[1]
    assert out.shape == (6, 64)  # size_m * top_k rows
    assert out.dtype == torch.half
    assert torch.all(out == 0)


def test_without_the_hint_the_buffer_is_passed_through(monkeypatch):
    c = torch.full((6, 64), 7.0, dtype=torch.half)
    args = _call(monkeypatch, c_or_none=c, zero_inactive_output=False)

    assert args[1] is c
    assert torch.all(c == 7.0), "nothing may be zeroed when the pin did not ask"


def test_the_sorted_hints_are_accepted(monkeypatch):
    c = torch.full((6, 64), 7.0, dtype=torch.half)
    # The SM70 kernel reads expert_ids directly; the hints only select the
    # pin's sorted specialization, so accepting them must change nothing.
    args = _call(
        monkeypatch,
        c_or_none=c,
        zero_inactive_output=False,
        expert_ids_are_sorted=True,
        has_inactive_prefix_count=True,
    )

    assert args[1] is c
    assert torch.all(c == 7.0)
