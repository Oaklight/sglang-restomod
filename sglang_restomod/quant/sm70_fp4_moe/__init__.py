"""SM70 NVFP4 MoE through the marlin_v100 extension.

The checkpoint side loads at the pin once ``_install_fp4_min_capability``
lowers ``ModelOptFp4Config.get_min_capability`` to 70. What is missing upstream
is the expert GEMM: the Marlin kernel shipped with SGLang is an empty-body stub
below SM80, so routed experts silently compute nothing on Volta.

This component supplies the kernel from ``marlin_v100`` (see PROVENANCE.md) and
bridges three upstream hooks:

* the load-time repack and scale encoding (``prepare_moe_nvfp4_layer_for_marlin``);
* the expert GEMM call (``moe_wna16_marlin_gemm``);
* nothing else: launch with ``--moe-runner-backend marlin`` and
  ``--quantization modelopt_fp4``.

Requirements:

* ``SM70_MARLIN_DIR`` points at the built ``_C*.so`` and ``_moe_C*.so``;
* FP16 activations, ``group_size=16``, gated experts, no expert bias;
* K divisible by 16 and N by 64 per GEMM (TP4 gives N=320 and 2560 here).

The standalone decode kernel in ``attention/v100/kernels`` is not used: the
Marlin kernel covers decode and prefill with one layout.

Issue: sglang-restomod#95.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Install the SM70 NVFP4 hooks (libraries load on first use)."""
    from .method import register as _install

    _install()
