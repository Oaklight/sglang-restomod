#!/usr/bin/env python3
"""Update UPSTREAM_PIN.toml and v100_plus/provenance.json together."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = ROOT / "UPSTREAM_PIN.toml"
PROV = ROOT / "v100_plus" / "provenance.json"
BASELINE = "heislera763/sglang-v100-plus@595b5f01d260320966060f734efabe2e379ebdce"


def resolve(ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="commit or tag to pin")
    parser.add_argument("--tag", default="", help="release tag name, if any")
    args = parser.parse_args()

    commit = resolve(args.ref)
    header = PIN.read_text().split("[upstream]", 1)[0].rstrip()
    body = (
        "[upstream]\n"
        'repo = "sgl-project/sglang"\n'
        'kind = "commit"\n'
        f'ref = "{commit}"\n'
        f'tag = "{args.tag}"\n'
        f'pinned_at = "{datetime.date.today().isoformat()}"\n'
        f'baseline = "{BASELINE}"\n'
    )
    PIN.write_text(f"{header}\n\n{body}")

    provenance = json.loads(PROV.read_text())
    provenance["current_upstream"] = commit
    PROV.write_text(json.dumps(provenance, indent=1) + "\n")

    print(f"pinned {args.tag or args.ref} -> {commit}")
    print(f"updated {PIN.relative_to(ROOT)} and {PROV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
