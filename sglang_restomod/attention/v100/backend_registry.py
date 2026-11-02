"""Register this component's attention backends with the pinned engine.

The pin publishes two extension points for exactly this purpose, so no patch
against the engine is required:

* ``sglang.srt.layers.attention.attention_registry.register_attention_backend``
  puts a factory into ``ATTENTION_BACKENDS``; the runner looks the name up at
  backend-construction time.
* ``sglang.srt.arg_groups.choices.add_attention_backend_choices`` extends
  ``ATTENTION_BACKEND_CHOICES``, the list every ``--attention-backend`` argument
  holds a reference to. The file documents it as the hook for out-of-tree
  platforms, and ``load_plugins()`` runs before ``ServerArgs`` construction, so
  extending it from a component reaches argparse in time.

Both operations are idempotent here because the plugin entry point can be
invoked more than once in one process (worker spawn, tests).

The factories import the backend lazily: ``register()`` must stay cheap, and
importing the backend pulls in torch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, MutableSequence

logger = logging.getLogger(__name__)

FLASH_ATTN_V100 = "tilelang_fa_v100"
QSA_V100 = "qsa_v100"

BACKEND_NAMES = (FLASH_ATTN_V100, QSA_V100)


def _make_flash_attn_v100(runner):
    from .flash_attn_v100_backend import FlashAttnV100Backend

    return FlashAttnV100Backend(runner)


def _make_qsa_v100(runner):
    from .qwen_sparse_attn_backend import QwenSparseAttnBackend

    return QwenSparseAttnBackend(runner)


def _factories() -> dict[str, Callable]:
    return {
        FLASH_ATTN_V100: _make_flash_attn_v100,
        QSA_V100: _make_qsa_v100,
    }


def build_backend(name: str, runner):
    """Construct a registered backend for ``runner``.

    The draft factory needs this: it resolves backend names through a map of
    its own, and that map cannot hold this component's names.
    """
    return _factories()[name](runner)


def _engine_hooks():
    from sglang.srt.arg_groups import choices
    from sglang.srt.layers.attention import attention_registry

    return (
        attention_registry.ATTENTION_BACKENDS,
        attention_registry.register_attention_backend,
        choices.ATTENTION_BACKEND_CHOICES,
        choices.add_attention_backend_choices,
    )


def install_qsa_swap() -> bool:
    """Replace the engine's own QSA backend class with this component's.

    ``attention_registry`` hard-codes its ``QwenSparseAttnBackend`` for hybrid
    GDN models whose full-attention layers are QSA (see the ``is_qwen_qsa``
    branch), so ``--attention-backend`` never reaches the registry for that
    model. The name is resolved when the backend is constructed, which is after
    plugin load, so rebinding the module attribute is enough.

    The engine class needs flash_attn for its packed varlen fallback; this
    component carries the sm70 CUDA decode path instead.
    """
    try:
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as pin_qsa

        from .qwen_sparse_attn_backend import QwenSparseAttnBackend
    except Exception as exc:  # noqa: BLE001 - report, never abort loading
        logger.warning("attention_v100: QSA backend swap skipped: %s", exc)
        return False

    if getattr(pin_qsa.QwenSparseAttnBackend, "_restomod_v100", False):
        return False
    pin_qsa._restomod_original_backend = pin_qsa.QwenSparseAttnBackend
    pin_qsa.QwenSparseAttnBackend = QwenSparseAttnBackend
    QwenSparseAttnBackend._restomod_v100 = True
    logger.info("attention_v100: replaced the engine QSA backend")
    return True


def install(
    *,
    registry: MutableSequence[str] | Mapping[str, Callable] | None = None,
    register: Callable[[str], Callable] | None = None,
    choices: Iterable[str] | None = None,
    extend_choices: Callable[[Iterable[str]], None] | None = None,
) -> bool:
    """Register the backends. Returns True if anything changed.

    The keywords exist so tests can inject fakes; production callers pass none
    and the engine hooks are imported on the spot.
    """
    if registry is None or register is None or choices is None or extend_choices is None:
        (
            engine_registry,
            engine_register,
            engine_choices,
            engine_extend,
        ) = _engine_hooks()
        registry = engine_registry if registry is None else registry
        register = engine_register if register is None else register
        choices = engine_choices if choices is None else choices
        extend_choices = engine_extend if extend_choices is None else extend_choices

    changed = False
    missing_choices = []

    for name, factory in _factories().items():
        if name not in registry:
            register(name)(factory)
            changed = True
        if name not in choices:
            missing_choices.append(name)

    if missing_choices:
        extend_choices(missing_choices)
        changed = True

    if changed:
        logger.info("attention_v100: registered backend(s) %s", ", ".join(BACKEND_NAMES))
    return changed
