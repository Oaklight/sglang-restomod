#!/usr/bin/env python3
"""Start each engine in turn and record the stage it reaches.

One command per engine, one JSON result per engine, and no state shared
between them: each runs in its own process on its own port, and the next one
starts only after the previous process is gone.

Inputs are a JSON file the caller supplies, so no machine path is committed:

    {
      "engines": [
        {"name": "vanilla", "python": "/path/to/python",
         "env": {"PYTHONPATH": "/path/to/tree"}}
      ],
      "launch": {"model_path": "/path/to/checkpoint", "tp_size": 2,
                 "port_base": 12100, "context_length": 8192,
                 "startup_timeout_s": 600},
      "forward": {"prompt": "The capital of France is", "max_new_tokens": 8}
    }

Stages, in order: ``import`` (the engine loads), ``health`` (the HTTP server
answers), ``forward`` (one generation returns text). The result records the
last stage reached and, on failure, one redacted line from the engine log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Rules shared with the evidence redactor: a path under a home or site mount is
# replaced before anything is written.
_PATH_ROOTS = (
    "home",
    "homes",
    "Users",
    "root",
    "nfs",
    "scratch",
    "gpfs",
    "lustre",
    "beegfs",
    "raid",
    "workspace",
    "mnt",
    "media",
)
_PATH_ALT = "|".join(_PATH_ROOTS)
_PATH_RE = re.compile(
    r"(?<![\w])/(?:" + _PATH_ALT + r")/[^\s\"'`,;:)\]]*"
)


def redact(text: str) -> str:
    return _PATH_RE.sub(lambda m: "<path>/" + m.group(0).rstrip("/").split("/")[-1], text)


def http_get(url: str, timeout: float = 3.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tail(path: str, limit: int = 4000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


_GENERIC = ("sigquit", "child failed", "no health response")


def first_error(log: str) -> str:
    """The most specific line a failure left behind.

    A dead scheduler reports ``signum=None ... child failed`` after the real
    exception, so a plain reverse scan returns the notice instead of the
    cause. Prefer the last exception line of the last traceback, then any
    line that names an error and is not one of those notices.
    """
    marker = "Traceback (most recent call last)"
    if marker in log:
        block = log[log.rfind(marker) :]
        # The scheduler logs more after the exception (signals, teardown);
        # those lines start with a timestamp, the exception does not.
        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("[")
        ]
        if lines:
            return redact(lines[-1])[:300]
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    for line in reversed(lines):
        low = line.lower()
        if any(token in low for token in _GENERIC):
            continue
        if "error" in low or "exception" in low or "failed" in low:
            return redact(line)[:300]
    return redact(lines[-1])[:300] if lines else ""


def run_engine(spec: dict, launch: dict, forward: dict, workdir: str) -> dict:
    name = spec["name"]
    port = int(launch["port_base"]) + spec.get("port_offset", 0)
    log_path = os.path.join(workdir, f"{name}.log")
    env = dict(os.environ)
    env.update({k: v for k, v in spec.get("env", {}).items()})
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        spec["python"],
        "-m",
        "sglang.launch_server",
        "--model-path",
        launch["model_path"],
        "--tp-size",
        str(launch.get("tp_size", 2)),
        "--port",
        str(port),
        "--context-length",
        str(launch.get("context_length", 8192)),
        "--dtype",
        launch.get("dtype", "float16"),
    ]
    command += spec.get("args", [])

    result = {"name": name, "stage": "import", "ok": False, "port": port}
    started = time.time()
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        deadline = started + float(launch.get("startup_timeout_s", 600))
        healthy = False
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if http_get(f"http://127.0.0.1:{port}/health") == 200:
                healthy = True
                break
            time.sleep(3)
        result["startup_s"] = round(time.time() - started, 1)
        if not healthy:
            result["error"] = first_error(tail(log_path)) or "no health response"
            return result

        result["stage"] = "health"
        payload = {
            "text": forward["prompt"],
            "sampling_params": {
                "max_new_tokens": forward.get("max_new_tokens", 8),
                "temperature": 0,
            },
        }
        try:
            reply = http_post(f"http://127.0.0.1:{port}/generate", payload)
        except Exception as exc:  # noqa: BLE001 - the stage is the result
            result["error"] = redact(f"{type(exc).__name__}: {exc}")[:300]
            return result
        text = reply.get("text", "")
        result["stage"] = "forward"
        result["ok"] = bool(text.strip())
        result["completion_chars"] = len(text)
        if not text.strip():
            result["error"] = "empty completion"
        return result
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=60)
        except Exception:  # noqa: BLE001 - best effort teardown
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", required=True, help="JSON file described above")
    parser.add_argument("--out", required=True, help="where to write the results JSON")
    parser.add_argument("--only", help="comma list of engine names to run")
    args = parser.parse_args(argv)

    with open(args.engines, encoding="utf-8") as fh:
        config = json.load(fh)
    engines = config["engines"]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        engines = [e for e in engines if e["name"] in wanted]

    workdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(workdir, exist_ok=True)

    results = []
    for index, spec in enumerate(engines):
        spec.setdefault("port_offset", index)
        print(f"== {spec['name']}", file=sys.stderr)
        results.append(run_engine(spec, config["launch"], config["forward"], workdir))
        print(
            f"   stage={results[-1]['stage']} ok={results[-1]['ok']}"
            f" {results[-1].get('error', '')}",
            file=sys.stderr,
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
