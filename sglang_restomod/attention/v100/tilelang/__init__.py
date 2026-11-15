# Vendored from dg1kjd/sglang-sxm2 @ 73abfe1 (python/sglang/srt/layers/attention/tilelang_fa_v100/__init__.py),
# with the Oaklight/lambda-v100-lab SM70 patches applied. See PROVENANCE.md.
"""TileLang FlashAttention V100 (SM70) — vendored from GooseLLM's
3rdparty/tilelang-fa-v100. Only the paged prefill path is exposed; decode stays
on sglang's Triton backend. Adapted for sglang's tilelang 0.1.8
(T.GemmWarpPolicy accessor, decorator-form jit, BaseKernelAdapter patch).
"""

__all__ = ["gather_fp8_paged_kv", "grouped_decode_forward", "paged_forward"]


def __getattr__(name):
    """Import the adapter only when a public name is requested.

    The adapter pulls torch, tilelang and the kernel builders; importing this
    package must not (golden rule 1). ``from ... import paged_forward`` still
    works because Python falls back to this hook.
    """
    if name in __all__:
        from . import _paged_adapter

        return getattr(_paged_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
