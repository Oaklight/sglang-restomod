"""CPU tests for the SM70 TileLang GDN prefill hook."""

from __future__ import annotations

import pytest

from sglang_restomod.linear_attn_sm70 import gdn_tilelang


class _Kernel:
    """Stand-in for the engine's ``TritonGDNKernel``."""

    def __init__(self):
        self.calls = []

    def extend(self, q, k, v, g, beta, **kwargs):
        self.calls.append((q, k, v, g, beta, kwargs))
        return "triton"


def _installed():
    cls = type("K", (_Kernel,), {})
    assert gdn_tilelang.install_on(cls) is True
    return cls


def test_install_is_idempotent():
    cls = _installed()
    assert gdn_tilelang.install_on(cls) is False


@pytest.fixture
def guard_ready(monkeypatch):
    """The guard install imports the engine; a CPU test starts past that."""
    monkeypatch.setattr(gdn_tilelang, "_GUARD_INSTALLED", True)


def test_tilelang_result_wins(monkeypatch, guard_ready):
    cls = _installed()
    sentinel = ("tilelang", None, None)
    monkeypatch.setattr(gdn_tilelang, "tilelang_extend", lambda *a, **kw: sentinel)

    kernel = cls()
    assert kernel.extend("q", "k", "v", "g", "b") is sentinel
    assert kernel.calls == []


def test_fallback_reaches_the_engine_kernel(monkeypatch):
    cls = _installed()
    monkeypatch.setattr(gdn_tilelang, "tilelang_extend", lambda *a, **kw: None)

    kernel = cls()
    assert kernel.extend("q", "k", "v", "g", "b", ssm_states="s") == "triton"
    assert kernel.calls == [("q", "k", "v", "g", "b", {"ssm_states": "s"})]


@pytest.mark.parametrize(
    "extra",
    [
        {"store_checkpoints": True},
        {"state_checkpoint_cu_starts": object()},
        {"num_state_checkpoints": 3},
        {"track_state": object()},
        {"output": object()},
    ],
)
def test_engine_only_features_fall_back(extra):
    assert gdn_tilelang._needs_engine_features(extra) is True
    assert gdn_tilelang.tilelang_extend(
        "q", "k", "v", "g", "b",
        ssm_states="s", cache_indices="c", query_start_loc="l", **extra,
    ) is None


def test_plain_prefill_is_not_diverted():
    assert gdn_tilelang._needs_engine_features({}) is False
    assert gdn_tilelang._needs_engine_features(
        {"layer_id": 3, "extend_prefix_lens": None}
    ) is False


def test_gate_is_promoted_to_a_batch_dim():
    torch = pytest.importorskip("torch")
    gate = torch.zeros(7, 3)
    assert gdn_tilelang._unsqueeze_gate(gate).shape == (1, 7, 3)
    already = torch.zeros(1, 7, 3)
    assert gdn_tilelang._unsqueeze_gate(already).shape == (1, 7, 3)


def test_min_tokens_is_validated(monkeypatch):
    monkeypatch.delenv(gdn_tilelang.MIN_TOKENS_ENV, raising=False)
    assert gdn_tilelang._min_tokens() == gdn_tilelang.DEFAULT_MIN_TOKENS
    monkeypatch.setenv(gdn_tilelang.MIN_TOKENS_ENV, "1024")
    assert gdn_tilelang._min_tokens() == 1024
    for bad in ("-1", "many"):
        monkeypatch.setenv(gdn_tilelang.MIN_TOKENS_ENV, bad)
        with pytest.raises(ValueError, match=gdn_tilelang.MIN_TOKENS_ENV):
            gdn_tilelang._min_tokens()


def test_unsupported_shapes_are_refused(guard_ready):
    torch = pytest.importorskip("torch")
    q = torch.zeros(1, 4096, 8, 128, dtype=torch.float16)
    k = torch.zeros(1, 4096, 8, 128, dtype=torch.float16)
    v = torch.zeros(1, 4096, 24, 128, dtype=torch.float16)
    ssm = torch.zeros(4, 24, 128, 128, dtype=torch.float16)

    # No CUDA in this test, so every call is refused regardless of shape.
    assert gdn_tilelang._supported(q, k, v, ssm, True) is False
    assert gdn_tilelang.tilelang_extend(
        q, k, v, q[:, :, :, 0].squeeze(-1), q[:, :, :, 0].squeeze(-1),
        ssm_states=ssm, cache_indices=torch.zeros(1), query_start_loc=torch.zeros(2),
    ) is None


class _Idx:
    """Stand-in for an index tensor; only ``numel`` is read."""

    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


def test_tracking_needs_a_mask_and_rows():
    class Meta:
        has_mamba_track_mask = False
        track_ssm_h_src = _Idx(3)

    class Backend:
        forward_metadata = Meta()

    assert gdn_tilelang._tracking_needed(Backend()) is False

    Meta.has_mamba_track_mask = True
    assert gdn_tilelang._tracking_needed(Backend()) is True
    Meta.track_ssm_h_src = _Idx(0)
    assert gdn_tilelang._tracking_needed(Backend()) is False
    assert gdn_tilelang._tracking_needed(object()) is False


def test_snapshot_capable_kernels_do_not_need_h():
    class Meta:
        has_mamba_track_mask = True
        track_ssm_h_src = _Idx(2)

    class Kernel:
        supports_track_state_snapshot = True

    class Dispatcher:
        extend_kernel = Kernel()

    class Backend:
        forward_metadata = Meta()
        kernel_dispatcher = Dispatcher()

    assert gdn_tilelang._tracking_needed(Backend()) is False
    Kernel.supports_track_state_snapshot = False
    assert gdn_tilelang._tracking_needed(Backend()) is True


def test_track_guard_scopes_the_flag():
    class Meta:
        has_mamba_track_mask = True
        track_ssm_h_src = _Idx(2)

    class Backend:
        forward_metadata = Meta()

        def forward_extend(self, *args, **kwargs):
            return gdn_tilelang._tracking_active()

    assert gdn_tilelang.install_track_guard(Backend) is True
    assert gdn_tilelang.install_track_guard(Backend) is False
    assert Backend().forward_extend() is True
    assert gdn_tilelang._tracking_active() is False

    Meta.has_mamba_track_mask = False
    assert Backend().forward_extend() is False


def test_track_guard_clears_on_exception():
    class Meta:
        has_mamba_track_mask = True
        track_ssm_h_src = _Idx(2)

    class Backend:
        forward_metadata = Meta()

        def forward_extend(self, *args, **kwargs):
            raise RuntimeError("boom")

    gdn_tilelang.install_track_guard(Backend)
    with pytest.raises(RuntimeError, match="boom"):
        Backend().forward_extend()
    assert gdn_tilelang._tracking_active() is False
