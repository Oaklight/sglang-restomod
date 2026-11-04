#!/usr/bin/env python3
"""Emit a machine-independent smoke summary.

The summary holds the capability tuple, counts and timings, and nothing else.
No host name, user name, path, serial number or device UUID is read, so none
can reach the output. That is the point: a summary is safe to paste into an
issue.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def _median_ms(fn, reps: int = 3) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return round(statistics.median(times), 3)


def build_summary(torch_module=None) -> dict:
    summary: dict = {
        "schema": 1,
        "torch": None,
        "cuda": None,
        "device_count": 0,
        "devices": [],
        "timings_ms": {},
    }
    if torch_module is None:
        try:
            import torch as torch_module  # noqa: PLC0415
        except Exception:
            summary["error"] = "torch unavailable"
            return summary

    summary["torch"] = getattr(torch_module, "__version__", None)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        summary["error"] = "cuda unavailable"
        return summary

    summary["cuda"] = getattr(torch_module.version, "cuda", None)
    count = int(cuda.device_count())
    summary["device_count"] = count
    for index in range(count):
        props = cuda.get_device_properties(index)
        summary["devices"].append(
            {
                "index": index,
                "name": props.name,
                "capability": list(cuda.get_device_capability(index)),
                "memory_mb": int(props.total_memory // (1024 * 1024)),
            }
        )

    if count:
        device = cuda.current_device()
        a = torch_module.randn(1024, 1024, device=device, dtype=torch_module.float16)
        b = torch_module.randn(1024, 1024, device=device, dtype=torch_module.float16)

        def matmul() -> None:
            a @ b

        def copy() -> None:
            a.clone()

        cuda.synchronize()
        summary["timings_ms"]["matmul_1024_fp16"] = _median_ms(matmul)
        summary["timings_ms"]["copy_1024_fp16"] = _median_ms(copy)
        cuda.synchronize()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", help="write the summary here as well as stdout")
    args = parser.parse_args(argv)

    payload = json.dumps(build_summary(), indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
