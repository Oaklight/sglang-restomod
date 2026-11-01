"""Registration contract for the V100 attention backend.

The engine hooks are injected so these run without sglang or a GPU.
"""

from __future__ import annotations

import sys

from sglang_restomod.attention.v100 import backend_registry


def _fake_engine():
    registry: dict = {}
    choices: list[str] = []

    def register(name):
        def decorator(fn):
            registry[name] = fn
            return fn

        return decorator

    return registry, register, choices, choices.extend


def test_install_registers_backend_and_choice():
    registry, register, choices, extend = _fake_engine()

    changed = backend_registry.install(
        registry=registry, register=register, choices=choices, extend_choices=extend
    )

    assert changed is True
    assert backend_registry.FLASH_ATTN_V100 in registry
    assert backend_registry.FLASH_ATTN_V100 in choices


def test_install_is_idempotent():
    registry, register, choices, extend = _fake_engine()
    kwargs = dict(registry=registry, register=register, choices=choices, extend_choices=extend)

    backend_registry.install(**kwargs)
    changed = backend_registry.install(**kwargs)

    assert changed is False
    assert choices.count(backend_registry.FLASH_ATTN_V100) == 1


def test_factory_does_not_import_the_backend_eagerly():
    registry, register, choices, extend = _fake_engine()
    backend_registry.install(
        registry=registry, register=register, choices=choices, extend_choices=extend
    )

    module = backend_registry.__name__.rsplit(".", 1)[0] + ".flash_attn_v100_backend"
    assert module not in sys.modules
