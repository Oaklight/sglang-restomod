#!/usr/bin/env python3
"""Verify an SM70 host end to end and write a machine-independent summary.

Exit 0 only when every check passes; otherwise print one line on stderr and
exit 1. The summary goes to ``--json-out`` and holds the capability tuple,
counts and timings. Nothing reads a host name, user name, path, serial number
or device UUID, so none can reach the file.

Inputs come from flags or environment variables the caller sets; no default
points at a machine:

    --json-out PATH        $SGLANG_SMOKE_JSON_OUT
    --upstream PATH        $SGLANG_SMOKE_UPSTREAM
    --components LIST      $SGLANG_RESTOMOD_COMPONENTS
    --engine-config PATH   $SGLANG_SMOKE_ENGINE_CONFIG

``--engine-config`` is a JSON object passed to ``Engine(**config)``. It is the
only way to name a checkpoint and its launch flags, so the summary never has to
describe them.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SmokeError(RuntimeError):
    """One check failed; the message is the one-line reason."""


def _redact(text: str) -> str:
    """Apply the evidence redaction rules to one line."""
    spec = importlib.util.spec_from_file_location(
        "redact_evidence", os.path.join(ROOT, "scripts", "redact_evidence.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.redact_text(text, [], {})


@contextlib.contextmanager
def _captured():
    """Send fd 1 and 2 to a scratch file for the duration of the block.

    Importing the engine modules logs their install path on stdout. A public
    job log is part of the attack surface, so that output is dropped here.
    """
    saved = (os.dup(1), os.dup(2))
    with tempfile.TemporaryFile() as scratch:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(scratch.fileno(), 1)
        os.dup2(scratch.fileno(), 2)
        try:
            yield scratch
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def check_cuda(torch) -> dict:
    if not torch.cuda.is_available():
        raise SmokeError("torch.cuda.is_available() is false")
    count = int(torch.cuda.device_count())
    if count < 1:
        raise SmokeError("torch.cuda.device_count() is 0")
    return {"device_count": count}


def check_capability(torch) -> dict:
    capability = list(torch.cuda.get_device_capability(0))
    if tuple(capability) != (7, 0):
        raise SmokeError(f"compute capability {tuple(capability)}, want (7, 0)")
    return {"capability": capability, "device_name": torch.cuda.get_device_name(0)}


def check_plugin(components: str) -> dict:
    if not components:
        raise SmokeError("--components is required for the plugin check")
    # The plugin reads its selection from the environment, which is also how a
    # launch selects components. Setting it here proves the real path.
    os.environ["SGLANG_RESTOMOD_COMPONENTS"] = components
    groups = {ep.group for ep in importlib.metadata.entry_points()}
    if "sglang.srt.plugins" not in groups:
        raise SmokeError("entry point group sglang.srt.plugins is not installed")
    matches = [
        ep
        for ep in importlib.metadata.entry_points(group="sglang.srt.plugins")
        if ep.name == "restomod"
    ]
    if not matches:
        raise SmokeError("entry point restomod is not registered")

    from sglang_restomod import overrides

    with _captured():
        matches[0].load()()
    installed = bool(getattr(overrides, "_INSTALLED", False))
    if not installed:
        raise SmokeError("sglang_restomod.register ran but installed no override")
    return {
        "entry_point": matches[0].value,
        "components": [c for c in components.split(",") if c],
        "overrides_installed": installed,
    }


def check_patches(upstream: str, python: str) -> dict:
    if not upstream:
        raise SmokeError("--upstream is required for the patch check")
    result = subprocess.run(
        [python, os.path.join(ROOT, "scripts", "check_patches.py"), "--source", upstream],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()
        raise SmokeError(
            _redact("patch queue: " + (reason[-1] if reason else "check failed"))
        )
    return {"returncode": result.returncode}


def check_dtype_guard(torch) -> dict:
    """A bfloat16 request must resolve to float16 before the model loads.

    torch no longer refuses bfloat16 on sm70: cuBLAS upcasts it, so a matmul
    succeeds and would be silently slow. The engine's own guard is therefore
    the only thing that keeps a bf16 default off this stack.
    """
    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.model_executor.model_runner_components import load_model_utils

    guard = getattr(load_model_utils, "maybe_downgrade_dtype_for_legacy_gpu", None)
    if guard is None:
        raise SmokeError("the engine has no maybe_downgrade_dtype_for_legacy_gpu")
    if not getattr(guard, "_restomod_sm70", False):
        raise SmokeError("the sm70 dtype guard is not the restomod override")

    observed: list[tuple] = []

    class _Context:
        def override(self, *args, **kwargs) -> None:
            observed.append((args, kwargs))

    original = runtime_context.get_context
    runtime_context.get_context = lambda: _Context()
    try:
        class _Config:
            dtype = torch.bfloat16

        config = _Config()
        guard(model_config=config)
    finally:
        runtime_context.get_context = original

    if config.dtype != torch.float16:
        raise SmokeError(f"a bfloat16 request resolved to {config.dtype}")
    return {"requested": "bfloat16", "resolved": "float16", "overrides": len(observed)}


def check_generation(config_path: str) -> dict:
    if not config_path:
        raise SmokeError("--engine-config is required for the generation check")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    if config.get("dtype") != "float16":
        raise SmokeError("engine config dtype must be float16; V100 has no bfloat16")

    # The engine names the checkpoint path on its own stdout, so it runs in a
    # child process whose output is captured and dropped. A failure reason is
    # reduced to one redacted line.
    with tempfile.TemporaryDirectory() as scratch:
        result_path = os.path.join(scratch, "result.json")
        child = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "gpu_smoke_gen.py"),
                "--config",
                config_path,
                "--result",
                result_path,
            ],
            capture_output=True,
            text=True,
        )
        if child.returncode != 0 or not os.path.exists(result_path):
            tail = (child.stderr or child.stdout).strip().splitlines()
            reason = tail[-1] if tail else f"child exited {child.returncode}"
            raise SmokeError(_redact("engine: " + reason))
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)

    if result.get("completion_chars", 0) <= 0:
        raise SmokeError("the generation loop returned an empty completion")
    result["dtype"] = config["dtype"]
    return result


def build_summary(args, summary: dict | None = None, torch=None) -> dict:
    summary = summary if summary is not None else {
        "schema": 1,
        "ok": False,
        "device_count": 0,
        "checks": {},
    }

    def record(name, fn, *fn_args):
        summary["checks"][name] = {"ok": False}
        started = time.perf_counter()
        try:
            detail = fn(*fn_args)
        except SmokeError:
            summary["error_check"] = name
            raise
        summary["checks"][name] = {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            **detail,
        }

    if torch is None:
        import torch  # noqa: PLC0415

    record("cuda", check_cuda, torch)
    summary["device_count"] = summary["checks"]["cuda"]["device_count"]
    record("capability", check_capability, torch)
    record("plugin", check_plugin, args.components)
    record("patches", check_patches, args.upstream, sys.executable)
    record("dtype-guard", check_dtype_guard, torch)
    record("generation", check_generation, args.engine_config)
    summary["ok"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out", default=_env("SGLANG_SMOKE_JSON_OUT"), help="summary path"
    )
    parser.add_argument("--upstream", default=_env("SGLANG_SMOKE_UPSTREAM"))
    parser.add_argument(
        "--components",
        default=_env("SGLANG_RESTOMOD_COMPONENTS"),
        help="component list the plugin should load",
    )
    parser.add_argument(
        "--engine-config", default=_env("SGLANG_SMOKE_ENGINE_CONFIG")
    )
    args = parser.parse_args(argv)

    if not args.json_out:
        print("error: --json-out is required", file=sys.stderr)
        return 2

    summary = {"schema": 1, "ok": False, "device_count": 0, "checks": {}}
    try:
        build_summary(args, summary)
    except SmokeError as exc:
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - one line is the whole contract
        reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = ""

    with open(args.json_out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if reason:
        print(f"FAILED: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
