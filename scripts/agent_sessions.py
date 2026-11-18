#!/usr/bin/env python3
"""Start and stop the restomod agent lanes through agentabi.

Each lane is one long-lived agent process, spawned by `agentabi.peers` into a
working directory of this repository. The lanes are *idle* unless a prompt is
given, so `start` costs nothing but a process; a task is explicit.

The fleet lives in its own agentabi state directory, so it never touches
peers that belong to other work on this machine.

    python scripts/agent_sessions.py list
    python scripts/agent_sessions.py start              # every lane, idle
    python scripts/agent_sessions.py start kernel attention   # two lanes
    python scripts/agent_sessions.py send kernel "read docs/status.md and continue"
    python scripts/agent_sessions.py status kernel
    python scripts/agent_sessions.py stop kernel
    python scripts/agent_sessions.py stop-all

`agentabi` must be importable:

    pip install agentabi
    # or, from a source checkout, with the checkout on the path:
    PYTHONPATH=<agentabi-checkout>/src python scripts/agent_sessions.py list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def _repo_root(start: Path) -> Path:
    """Return the main checkout, even when this script runs from a worktree.

    A worktree lives at ``<repo>/.claude/worktrees/<name>/`` and carries a
    ``.git`` *file*, so the nearest ``.git`` is not the repository root: the
    ``.claude`` directory has to be looked for first, across every ancestor.
    """
    parents = list(start.parents)
    for parent in parents:
        if parent.name == ".claude" and (parent / "worktrees").is_dir():
            return parent.parent
    for parent in parents:
        if (parent / ".git").exists():
            return parent
    return start.parents[1]


REPO = _repo_root(Path(__file__).resolve())
DEFAULT_STATE_DIR = Path.home() / ".local/state/agentabi/restomod-peers"

#: Default lane -> working directory, relative to the repository root.
#:
#: One lane per worktree, because two agents in one directory edit the same
#: files. A lane whose directory does not exist is reported and skipped, never
#: created; add a lane with ``git worktree add .claude/worktrees/<name>`` and
#: either extend this map or pass ``--map``.
#:
#: Note: this manages peers that agentabi *spawns*. An agent started by hand
#: (a `claude` session in a terminal) cannot be adopted — stop it yourself
#: first, then start the lane.
LANES: dict[str, str] = {
    "integrator": ".claude/worktrees/env-unify",
    "kernels": ".claude/worktrees/mtp-fp4-moe",
    "docs": ".claude/worktrees/strata-res",
}


def _client(state_dir: Path):
    """Return a PeerClient bound to this fleet's state directory."""
    try:
        from agentabi.peers.client import PeerClient
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "agentabi is not importable. Either `pip install agentabi`, or put a "
            "source checkout's src/ directory on PYTHONPATH and re-run.\n"
            f"({exc})"
        )
    return PeerClient(state_dir)


def _lane_dir(lane: str, extra: dict[str, str] | None = None) -> Path:
    """Resolve a lane's working directory, refusing an unknown lane."""
    mapping = {**LANES, **(extra or {})}
    if lane not in mapping:
        raise SystemExit(
            f"unknown lane {lane!r}; known: {', '.join(sorted(mapping))} "
            "(add one with --map NAME=PATH)"
        )
    return (REPO / mapping[lane]).resolve()


def _parse_map(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``--map NAME=PATH`` pairs into a lane override map."""
    out: dict[str, str] = {}
    for pair in pairs or []:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--map expects NAME=PATH, got {pair!r}")
        out[name] = path
    return out


def _live(client) -> dict[str, dict]:
    """Map lane name -> peer info for peers in this fleet's state dir."""
    out: dict[str, dict] = {}
    for info in client.request("list") or []:
        if isinstance(info, dict):
            out[str(info.get("name"))] = info
    return out


def cmd_list(client, args: argparse.Namespace) -> int:
    peers = _live(client)
    if not peers:
        print("no lanes running in", client.state_dir)
        return 0
    width = max(len(name) for name in peers)
    for name, info in sorted(peers.items()):
        print(
            f"{name:<{width}}  {info.get('status', '?'):<9}"
            f"  pid={info.get('pid') or '-'}"
            f"  session={info.get('session_id') or '-'}"
            f"  cwd={info.get('cwd', '?')}"
        )
    return 0


def cmd_status(client, args: argparse.Namespace) -> int:
    peers = _live(client)
    names = args.lanes or sorted(peers)
    for name in names:
        info = peers.get(name)
        print(json.dumps(info or {"name": name, "status": "absent"}, indent=2))
    return 0


def cmd_start(client, args: argparse.Namespace) -> int:
    extra = _parse_map(args.map)
    names = args.lanes or sorted({**LANES, **extra})
    peers = _live(client)
    started = 0
    for name in names:
        cwd = _lane_dir(name, extra)
        if not cwd.is_dir():
            print(f"skip {name}: {cwd} does not exist", file=sys.stderr)
            continue
        existing = peers.get(name)
        if existing and existing.get("status") == "running":
            print(f"skip {name}: already running (pid {existing.get('pid')})")
            continue
        if args.dry_run:
            print(f"would spawn {name} in {cwd} (prompt: {args.prompt or '-'})")
            started += 1
            continue
        params = {"name": name, "cwd": str(cwd)}
        if args.prompt:
            params["prompt"] = args.prompt
        if args.permission_mode:
            params["permission_mode"] = args.permission_mode
        result = client.request("spawn", params, timeout=args.timeout)
        print(
            f"spawned {name} pid={result.get('pid')} "
            f"session={result.get('session_id') or '-'} in {cwd}"
        )
        started += 1
    if not started:
        print("nothing started", file=sys.stderr)
        return 1
    return 0


def cmd_stop(client, args: argparse.Namespace) -> int:
    peers = _live(client)
    names = args.lanes or sorted(peers)
    stopped = 0
    for name in names:
        if name not in peers:
            if args.lanes:
                print(f"skip {name}: not running", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"would stop {name} (pid {peers[name].get('pid')})")
            stopped += 1
            continue
        client.request("stop", {"name": name}, timeout=args.timeout)
        print(f"stopped {name}")
        stopped += 1
    if not stopped:
        print("nothing stopped", file=sys.stderr)
        return 1
    return 0


def cmd_stop_all(client, args: argparse.Namespace) -> int:
    args.lanes = []
    peers = _live(client)
    if not peers:
        print("no lanes running")
        return 0
    args.lanes = sorted(peers)
    return cmd_stop(client, args)


def cmd_send(client, args: argparse.Namespace) -> int:
    peers = _live(client)
    if args.lane not in peers:
        raise SystemExit(f"lane {args.lane!r} is not running; see `list`")
    result = client.request(
        "send", {"name": args.lane, "text": args.text}, timeout=args.timeout
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_shutdown(client, args: argparse.Namespace) -> int:
    """Stop every lane and then the daemon itself."""
    cmd_stop_all(client, args)
    if args.dry_run:
        return 0
    client.stop_daemon()
    print("peer daemon stopped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_sessions.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=f"agentabi peer state directory (default {DEFAULT_STATE_DIR})",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would happen"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list running lanes")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="show one or more lanes as JSON")
    p_status.add_argument("lanes", nargs="*")
    p_status.set_defaults(func=cmd_status)

    p_start = sub.add_parser("start", help="spawn lanes (idle unless --prompt)")
    p_start.add_argument("lanes", nargs="*")
    p_start.add_argument("--prompt", help="initial task; omit to spawn idle")
    p_start.add_argument(
        "--permission-mode", help="passed to the agent backend (e.g. bypassPermissions)"
    )
    p_start.add_argument(
        "--map",
        action="append",
        metavar="NAME=PATH",
        help="add or override a lane's directory, relative to the repository root",
    )
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="stop lanes by name")
    p_stop.add_argument("lanes", nargs="*")
    p_stop.set_defaults(func=cmd_stop)

    p_stop_all = sub.add_parser("stop-all", help="stop every lane in this fleet")
    p_stop_all.set_defaults(func=cmd_stop_all)

    p_send = sub.add_parser("send", help="send a message to a lane")
    p_send.add_argument("lane")
    p_send.add_argument("text")
    p_send.set_defaults(func=cmd_send)

    p_shutdown = sub.add_parser(
        "shutdown", help="stop every lane, then stop the daemon"
    )
    p_shutdown.set_defaults(func=cmd_shutdown)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    state_dir = Path(os.path.expanduser(str(args.state_dir or DEFAULT_STATE_DIR)))
    client = _client(state_dir)
    return int(args.func(client, args))


if __name__ == "__main__":
    raise SystemExit(main())
