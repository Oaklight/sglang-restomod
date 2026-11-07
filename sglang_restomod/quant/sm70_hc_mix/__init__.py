"""Route the hyper-connection mixer through each linear's own forward.

``GatedResidual.mix`` calls ``F.linear(x, self.input_mix_weight_down.weight)``
on the raw tensor, so a quantized linear is bypassed: ``fp8_dense_quant``
replaces the weight and the module's ``forward``, but the mixer never calls
``forward``. On Flash-Next the pair is ``[320, 10240]`` and
``[10240, 320]`` per rank, 97 + 97 calls per decode step. The 320-wide axis
misses the block-128 rule, so the only route is a zero-padded conversion,
whose forward pads the input and slices the output -- all of which the mixer
skips. The failure is not a wrong number: decode CUDA-graph capture dies on
the fp8 weight reaching an fp16 ``F.linear``.

This component wraps ``GatedResidual.__init__`` and, on an instance whose two
mixer linears are already converted, replaces the mixer's compute step with
one that calls those modules. Unconverted instances keep the upstream path, so
selecting the component on a launch that converts nothing changes nothing.

It is opt-in by component name (``hc_mix_fp8``), off by default. The install
asserts the class layout it depends on and raises when a pin bump changes it,
so the failure is a startup error rather than a silent return to ``F.linear``.
The server log carries one marker line when it takes effect:
``sglang-restomod: hc mixer override active``.
"""

from __future__ import annotations

import inspect
import logging
from importlib import import_module

logger = logging.getLogger(__name__)

BLOCK = 128
MODULE = "sglang.srt.layers.hyperconnection"
MARKER = "sglang-restomod: hc mixer override active"

# The upstream layout this module depends on. A pin bump that renames or
# restructures any of these is caught at install time.
_REQUIRED_IN_INIT = (
    "self.input_mix_weight_down = nn.Linear(",
    "self.input_mix_weight_up = nn.Linear(",
    "self._mix_compute = ",
    "F.linear(hyper_input_normed, input_mix_weight_down)",
)

_installed = 0
_logged = False


def _converted(module) -> bool:
    """Whether ``fp8_dense_quant`` has replaced this linear's path."""
    if getattr(module, "sm70_fp8_padded_in_features", None) is not None:
        return True
    return getattr(module, "weight_block_size", None) == [BLOCK, BLOCK]


def _check_class(cls) -> None:
    """Fail when the pin's ``GatedResidual`` no longer has the shape we patch."""
    if not hasattr(cls, "mix"):
        raise RuntimeError(
            "hc_mix_fp8: GatedResidual has no mix(); the pin changed, refusing to "
            "install a mixer override"
        )
    source = inspect.getsource(cls.__init__)
    missing = [token for token in _REQUIRED_IN_INIT if token not in source]
    if missing:
        raise RuntimeError(
            "hc_mix_fp8: GatedResidual.__init__ no longer matches the layout this "
            f"override patches (missing: {missing}); a pin bump changed the "
            "hyper-connection mixer"
        )


def _install_instance(instance) -> bool:
    """Patch one GatedResidual; return whether the mixer now uses its modules."""
    global _installed, _logged

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if getattr(instance, "_restomod_hc_mix", False):
        return False
    down = getattr(instance, "input_mix_weight_down", None)
    up = getattr(instance, "input_mix_weight_up", None)
    if not isinstance(down, nn.Linear) and not _converted(down):
        raise RuntimeError(
            "hc_mix_fp8: input_mix_weight_down is "
            f"{type(down).__name__}, not a linear"
        )
    if not isinstance(up, nn.Linear) and not _converted(up):
        raise RuntimeError(
            f"hc_mix_fp8: input_mix_weight_up is {type(up).__name__}, not a linear"
        )
    hidden = int(instance.config.hidden_size)
    hc = int(instance.hc_count)
    lowrank = int(instance.config.hc_lowrank)

    def _shape(module) -> tuple[int, int]:
        # A converted linear's ``weight`` is packed by the kernel; the module
        # records the logical output width and the padded input width.
        if _converted(module):
            return int(module.sm70_fp8_out_features), int(
                module.sm70_fp8_padded_in_features
            )
        return tuple(int(d) for d in module.weight.shape)

    def _fits(module, want_n: int, want_k: int) -> bool:
        got_n, got_k = _shape(module)
        if got_n != want_n:
            return False
        if not _converted(module):
            return got_k == want_k
        # Zero padding leaves K at or above the logical width, on the block.
        return got_k == want_k or (got_k > want_k and got_k % BLOCK == 0)

    if not _fits(down, lowrank, hc * hidden):
        raise RuntimeError(
            f"hc_mix_fp8: input_mix_weight_down is {_shape(down)}, expected "
            f"{(lowrank, hc * hidden)}"
        )
    if not _fits(up, hc * hidden, lowrank):
        raise RuntimeError(
            f"hc_mix_fp8: input_mix_weight_up is {_shape(up)}, expected "
            f"{(hc * hidden, lowrank)}"
        )

    if not (_converted(down) and _converted(up)):
        # The launch converted nothing (or only one side): upstream behaviour.
        return False

    def _mix_compute(hyper_input_normed, _down_weight, _up_weight, hc_arg, hs_arg):
        weight = torch.sigmoid(up(F.silu(down(hyper_input_normed) / hc_arg)))
        weight = weight.unflatten(-1, (hc_arg, hs_arg))
        return (weight * hyper_input_normed.unflatten(-1, (hc_arg, hs_arg))).mean(
            dim=-2
        )

    instance._mix_compute = _mix_compute
    instance._restomod_hc_mix = True
    _installed += 1
    if not _logged:
        _logged = True
        logger.info("%s (%d x %d, low-rank %d)", MARKER, hc * hidden, hidden, lowrank)
    else:
        logger.debug("hc_mix_fp8: mixer routed through converted linears")
    return True


def install(module=None) -> int:
    """Wrap ``GatedResidual.__init__``; return 1 when the hook was installed."""
    module = module if module is not None else import_module(MODULE)
    cls = getattr(module, "GatedResidual", None)
    if cls is None:
        raise RuntimeError(f"hc_mix_fp8: {module.__name__} has no GatedResidual")
    original = cls.__init__
    if getattr(original, "_restomod_hc_mix", False):
        return 0
    _check_class(cls)

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _install_instance(self)

    __init__._restomod_hc_mix = True
    cls.__init__ = __init__
    return 1


def installed() -> int:
    """How many mixer instances the override has taken over."""
    return _installed


def register() -> None:
    """Component entry point."""
    install()
