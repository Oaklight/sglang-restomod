#!/usr/bin/env python3
"""Report the custom-allreduce state of one server log.

A custom-allreduce setup that fails to build does not fail the launch. The
server logs one warning, falls back to NCCL ring allreduce, and serves at a
plausible speed; #260 lost about 3.1 ms per decode step this way while every
number still looked like a regression in something else. The log is the only
place the fallback is visible.

At the pin (sglang v0.5.23, 297e54fd) there is no success message on the CUDA
path: ``parallel_state.py`` logs only the failure warning, and every
``[AR]`` line that reports a working setup is either AMD-only or
``logger.debug``. So this check can establish a failure and can never establish
a success. It reports which of the three states it reached rather than
implying more:

    custom_allreduce=failed               the failure marker is in the log
    custom_allreduce=failure-absent-only  no marker; the fallback is not
                                          excluded, and nothing confirms the
                                          setup worked
    custom_allreduce=not checked          no --server-log was supplied

Exit status: 0 for a recorded state that is not a failure, 1 for a failed
custom allreduce, 2 when the log it was given cannot be read. A wrong path is
not a state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Verbatim from sglang/srt/distributed/parallel_state.py, the
# `use_custom_allreduce` block at the pin above (line 515):
#     logger.warning(f"Setup Custom allreduce failed with {e}. To silence ...")
# The exception text that follows carries the build error, for example
# "Failed to build JIT module sgl_kernel_jit_cuda_ipc". Do not match that string
# on its own: other JIT modules fail with the same wording.
FAILURE_MARKER = "Setup Custom allreduce failed with"

NOT_CHECKED = "custom_allreduce=not checked"
FAILURE_ABSENT = (
    "custom_allreduce=failure-absent-only (no success marker exists at the pin; "
    "this does not confirm custom allreduce is active)"
)
FAILED = "custom_allreduce=failed"


def scan(log_path: Path) -> tuple[str, int]:
    """Return (state, first matching line number), or raise OSError."""
    with log_path.open("r", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if FAILURE_MARKER in line:
                return FAILED, number
    return FAILURE_ABSENT, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-log",
        default=None,
        help="server log to scan; omit to record that the check did not run",
    )
    parser.add_argument("--out", required=True, help="state file to write")
    args = parser.parse_args(argv)

    lines: list[str] = []
    status = 0

    if args.server_log is None:
        lines.append(NOT_CHECKED)
    else:
        log_path = Path(args.server_log)
        try:
            state, number = scan(log_path)
        except OSError as exc:
            print(f"error: cannot read --server-log {log_path}: {exc}", file=sys.stderr)
            return 2
        lines.append(state)
        lines.append(f"server_log={log_path}")
        lines.append(f"marker={FAILURE_MARKER!r}")
        if number:
            lines.append(f"line={number}")
            status = 1

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(lines[0])
    return status


if __name__ == "__main__":
    sys.exit(main())
