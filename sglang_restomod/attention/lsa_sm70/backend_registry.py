"""Register the LSA attention backend with the pinned engine.

The pin publishes two extension points for out-of-tree backends:
``attention_registry.register_attention_backend`` puts a factory into
``ATTENTION_BACKENDS`` and ``choices.add_attention_backend_choices`` extends the
``--attention-backend`` list. Both are idempotent; the plugin entry point can run
more than once in one process.

The factory imports the backend lazily, so ``register()`` stays cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, MutableSequence

logger = logging.getLogger(__name__)

LSA_SM70 = "lsa_sm70"


def _make_lsa_sm70(runner):
    from .backend import LsaSm70Backend

    return LsaSm70Backend(runner)


def _factories() -> dict[str, Callable]:
    return {LSA_SM70: _make_lsa_sm70}


def _engine_hooks():
    from sglang.srt.arg_groups import choices
    from sglang.srt.layers.attention import attention_registry

    return (
        attention_registry.ATTENTION_BACKENDS,
        attention_registry.register_attention_backend,
        choices.ATTENTION_BACKEND_CHOICES,
        choices.add_attention_backend_choices,
    )


def install(
    *,
    registry: MutableSequence[str] | Mapping[str, Callable] | None = None,
    register: Callable[[str], Callable] | None = None,
    choices: Iterable[str] | None = None,
    extend_choices: Callable[[Iterable[str]], None] | None = None,
) -> bool:
    """Register the backend. Returns True if anything changed.

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
        logger.info("lsa_sm70: registered backend %s", LSA_SM70)
    return changed
