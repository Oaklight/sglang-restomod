#!/usr/bin/env python3
"""Run one generation through the engine and write a small result file.

This is the child of ``scripts/gpu_smoke.py``'s generation check. It runs in
its own process so the engine's own stdout and stderr -- which name the
checkpoint path -- never reach the caller's output. Only the prompt length,
the completion length and the wall time go into the result file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON object for Engine(**config)")
    parser.add_argument("--result", required=True, help="where to write the result JSON")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)

    from sglang import Engine

    prompt = "The capital of France is"
    started = time.perf_counter()
    engine = Engine(**config)
    output = engine.generate(prompt, {"max_new_tokens": 8, "temperature": 0})
    elapsed = time.perf_counter() - started

    text = output.get("text", "") if isinstance(output, dict) else str(output)
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "prompt_chars": len(prompt),
                "completion_chars": len(text),
                "elapsed_s": round(elapsed, 3),
            },
            fh,
        )
    if not text.strip():
        print("empty completion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
