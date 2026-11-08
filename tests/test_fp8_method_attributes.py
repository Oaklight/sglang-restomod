"""The SM70 block-FP8 method must be built through its constructor.

``sm70_fp8_lm_head`` and ``sm70_fp8_dense_quant`` used to build
``Fp8LinearMethod`` with ``__new__`` and set two attributes. v0.5.23's
``process_weights_after_loading`` reads ``block_fp8_as_mxfp8``, which only the
constructor sets, so model init failed on every rank. The hook now constructs
through ``Fp8Config``; these tests keep the next pin bump from repeating it.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The attributes the called path reads. The runtime test needs them on the
# method the hook builds; the static test needs the pinned constructor to set
# them. Audit both lists when a test fails, not just the missing name.
METHOD_ATTRIBUTES = (
    "quant_config",
    "block_quant",
    "block_fp8_as_mxfp8",
    "weight_block_size",
    "use_mxfp8",
    "convert_mxfp8_to_block",
    "is_checkpoint_fp8_serialized",
    "use_marlin",
    "cutlass_fp8_supported",
    "use_aiter_fp8_per_token",
    "use_per_token_if_dynamic",
    "w8a8_block_fp8_linear",
    "mxfp8_dense_backend",
    "mxfp8_prefill_autotune_min_tokens",
    "w8a8_mxfp8_linear",
    "use_sm70_turbomind",  # set by the factory, read by the patched prepare
)


def test_the_factory_sets_every_attribute_the_prepare_path_reads():
    pytest.importorskip("sglang.srt.layers.quantization.fp8")

    from sglang_restomod.quant.sm70_fp8_dense import make_block_fp8_method

    method = make_block_fp8_method([128, 128])

    missing = [name for name in METHOD_ATTRIBUTES if not hasattr(method, name)]
    assert not missing, f"the method built by the hook is missing {missing}"


def _pinned_source() -> str | None:
    """Return the pinned ``fp8.py``, or None when upstream lags the pin."""
    pin = tomllib.loads((ROOT / "UPSTREAM_PIN.toml").read_text())
    expected = pin["upstream"]["commit"]
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "origin/upstream"],
        capture_output=True,
        text=True,
    )
    if head.returncode != 0 or head.stdout.strip() != expected:
        return None
    shown = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            "origin/upstream:python/sglang/srt/layers/quantization/fp8.py",
        ],
        capture_output=True,
        text=True,
    )
    return shown.stdout if shown.returncode == 0 else None


def _method_body(class_body: str, name: str) -> str:
    """The indented body of a method, tolerating a multi-line signature."""
    match = re.search(rf"^    def {name}\(", class_body, re.M)
    if match is None:
        return ""
    lines = class_body[match.start():].splitlines()
    end_of_signature = 0
    while end_of_signature < len(lines) and not lines[end_of_signature].rstrip().endswith(":"):
        end_of_signature += 1
    body: list[str] = []
    for line in lines[end_of_signature + 1:]:
        if line.strip() and not line.startswith("        "):
            break
        body.append(line)
    return "\n".join(body)


def test_the_pinned_constructor_sets_the_attributes_the_called_path_reads():
    source = _pinned_source()
    if source is None:
        pytest.skip("origin/upstream is not at the pinned commit; the pin cannot be checked")

    start = source.index("class Fp8LinearMethod(")
    end = source.index("\nclass ", start + 1)
    class_body = source[start:end]

    init = _method_body(class_body, "__init__")
    called = "\n".join(
        _method_body(class_body, name)
        for name in (
            "process_weights_after_loading",
            "process_weights_after_loading_block_quant",
            "apply",
        )
    )
    assert init and called, "the pinned Fp8LinearMethod shape changed; re-audit the hook"

    assigned = set(re.findall(r"self\.([A-Za-z_]\w*)\s*=", init))
    assigned |= set(re.findall(r"self\.([A-Za-z_]\w*)\s*=", called))
    read = set(re.findall(r"self\.([A-Za-z_]\w*)", called))
    methods = set(re.findall(r"^    def ([A-Za-z_]\w*)\(", class_body, re.M))
    required = read - methods - {"use_sm70_turbomind"}

    missing = sorted(required - assigned)
    assert not missing, (
        "the pinned constructor does not set attributes the called path reads: "
        f"{missing}. Set them in the SM70 factory, not one AttributeError at a time."
    )
