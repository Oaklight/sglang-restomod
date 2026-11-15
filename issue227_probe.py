#!/usr/bin/env python3
"""One greedy completion, saved for the #227 A/B output check."""
import argparse
import json
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:11436")
    ap.add_argument("--prompt", default=(
        "A distributed inference server must keep the decode step stable. "
        "In one paragraph, explain why memory bandwidth, not arithmetic, "
        "sets the token rate at batch size one."
    ))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = "http://" + args.host
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as resp:
        model = json.load(resp)["data"][0]["id"]
    payload = {
        "model": model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_k": 1,
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.load(resp)
    choice = (obj.get("choices") or [{}])[0]
    text = choice.get("text", "")
    record = {"model": model, "text": text, "usage": obj.get("usage"),
              "finish_reason": choice.get("finish_reason")}
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
