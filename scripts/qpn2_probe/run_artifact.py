#!/usr/bin/env python3
"""Collect one ``qpn2-run-v1`` artifact per arm for the P3/P4 guard (#224).

Drives a running SGLang server (``/generate``) with the guard's prompt set and
records, per prompt: the 128 greedy output token ids, the chosen token's
logprob at each position, the top-2 logprobs and token ids at each position,
and the speculative accept length. ``compare_runs.py`` consumes the result.

The server has to run with logprobs enabled (``--enable-custom-logit-processor``
is not needed; ``return_logprob`` on the request is). Each request asks for
``return_logprob: true`` and ``top_logprobs_num: 2``, greedy sampling with
``ignore_eos`` so every prompt yields exactly ``--max-new-tokens`` tokens.

Everything here is testable without the engine:

    # a local fake /generate that returns canned, well-formed responses
    python run_artifact.py collect --dry-run --out /tmp/a.json --layout marlin \
        --commit dry
    python run_artifact.py fake-server --port 18400 &
    python run_artifact.py collect --base-url http://127.0.0.1:18400 --out b.json ...
    python run_artifact.py selftest     # fake server + both arms + the comparator

On the real 4-card run::

    python run_artifact.py collect --base-url http://lambda4:30000 \
        --out /nfs/.../runA.json --layout marlin --commit "$(git rev-parse HEAD)" \
        --model Qwen3.8-Flash-Next --checkpoint "$CKPT" --tp 4 \
        --tokenizer "$TOKENIZER" --mtp
    # then the same against the QPN2 launch, with --layout qpn2 --qpn2-tensors both
    python compare_runs.py runA.json both=runB.json

Prompt set (guard section 3): contexts 76, 993, 3473, 7168; two contents
(repeated technical seed, and a word list); two reps each; 16 prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCHEMA = "qpn2-run-v1"
MAX_NEW_TOKENS = 128
TOP_LOGGROBS = 2
CONTEXTS = (76, 993, 3473, 7168)
REPS = 2
TECHNICAL_SEED = (
    "The kernel reads the KV cache once per split, so the traffic floor is the "
    "cache size itself. A split-KV attention kernel that re-reads the prefix "
    "per query head pays for the group size again and again. "
)
WORDS = (
    "harbor lantern meadow copper thistle ripple cavern granite swallow prairie "
    "orchard marble tunnel bramble compass drizzle hollow juniper kettle lattice"
)


# --- prompt construction ---------------------------------------------------

def build_prompt_texts(tokenizer=None, contexts=CONTEXTS, reps=REPS) -> list[dict]:
    """The guard's 16 prompts, identical across arms.

    With a tokenizer, each prompt is repeated seed text truncated to exactly the
    requested token count. Without one (dry run), the prompt text is synthetic
    and the token count is the requested number.
    """
    contents = {
        "a": TECHNICAL_SEED,
        "b": WORDS,
    }
    prompts = []
    for context in contexts:
        for name, seed in contents.items():
            for rep in range(reps):
                pid = f"ctx{context}-{name}-r{rep}"
                if tokenizer is None:
                    prompts.append(
                        {"id": pid, "context_tokens": context,
                         "prompt_tokens": context, "text": f"[dry {pid}] {seed}",
                         "seed_text": seed}
                    )
                    continue
                ids = tokenizer.encode(seed, add_special_tokens=False)
                if not ids:
                    raise SystemExit(f"seed {name!r} produced no tokens")
                full = (ids * ((context + len(ids) - 1) // len(ids)))[:context]
                prompts.append(
                    {"id": pid, "context_tokens": context, "prompt_tokens": len(full),
                     "text": tokenizer.decode(full), "seed_text": seed}
                )
    return prompts


# --- server request and response parsing -----------------------------------

def request_payload(prompt_text: str, max_new_tokens: int) -> dict:
    return {
        "text": prompt_text,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": TOP_LOGGROBS,
        "stream": False,
    }


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _logprob_pair(item) -> list:
    """``(logprob, token_id, text|None)`` -> ``[logprob, token_id]``."""
    if isinstance(item, dict):
        return [float(item["logprob"]), int(item.get("token_id", item.get("id")))]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [float(item[0]), int(item[1])]
    raise ValueError(f"unrecognised logprob entry {item!r}")


def parse_generate_response(response: dict, max_new_tokens: int) -> dict:
    meta = response.get("meta_info") or {}
    token_ids = response.get("output_ids") or meta.get("output_ids")
    if not token_ids:
        raise ValueError("response has no output_ids")
    chosen_raw = meta.get("output_token_logprobs") or []
    if not chosen_raw:
        raise ValueError("response has no output_token_logprobs; was return_logprob set?")
    chosen = [_logprob_pair(item)[0] for item in chosen_raw]
    top_raw = meta.get("output_top_logprobs") or []
    top2 = []
    for position in top_raw:
        entries = [_logprob_pair(item) for item in (position or [])]
        top2.append(entries[:TOP_LOGGROBS])

    count = min(len(token_ids), len(chosen), max_new_tokens)
    token_ids = [int(t) for t in token_ids[:count]]
    chosen = chosen[:count]
    top2 = top2[:count]

    result = {
        "token_ids": token_ids,
        "logprobs": chosen,
        "top2": top2,
        "completion_tokens": int(meta.get("completion_tokens", count)),
    }
    accept = meta.get("spec_accept_length")
    if accept is not None:
        result["accept_length"] = round(float(accept), 4)
    if meta.get("spec_verify_ct") is not None:
        result["verify_ct"] = int(meta["spec_verify_ct"])
    return result


def collect_prompt(base_url: str, prompt: dict, timeout: float, max_new_tokens: int) -> dict:
    response = post_json(f"{base_url.rstrip('/')}/generate",
                         request_payload(prompt["text"], max_new_tokens), timeout)
    record = {"id": prompt["id"], "context_tokens": prompt["context_tokens"],
              "prompt_tokens": prompt["prompt_tokens"]}
    record.update(parse_generate_response(response, max_new_tokens))
    return record


# --- artifact assembly -----------------------------------------------------

def build_artifact(launch: dict, records: list[dict], error=None) -> dict:
    return {
        "schema": SCHEMA,
        "launch": launch,
        "error": error,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts": records,
    }


def run_collect(args) -> int:
    prompts = build_prompt_texts(args.tokenizer, reps=args.reps)
    launch = {
        "commit": args.commit,
        "layout": args.layout,
        "qpn2_tensors": args.qpn2_tensors,
        "mtp": bool(args.mtp),
        "tp": args.tp,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "flags": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "top_logprobs_num": TOP_LOGGROBS,
        },
    }
    records = []
    error = None
    if args.dry_run:
        for prompt in prompts:
            record = {"id": prompt["id"], "context_tokens": prompt["context_tokens"],
                      "prompt_tokens": prompt["prompt_tokens"]}
            record.update(canned_generate(prompt["text"], args.max_new_tokens, args.fake_mode))
            records.append(record)
    else:
        try:
            for prompt in prompts:
                records.append(
                    collect_prompt(args.base_url, prompt, args.timeout, args.max_new_tokens)
                )
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"

    artifact = build_artifact(launch, records, error)
    Path(args.out).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    status = "error" if error else "ok"
    print(f"wrote {args.out}: {len(records)}/{len(prompts)} prompts, {status}")
    return 1 if error else 0


# --- canned responses (dry run and fake server) ----------------------------

def canned_generate(prompt_text: str, max_new_tokens: int = MAX_NEW_TOKENS,
                    mode: str = "match") -> dict:
    """Deterministic, schema-shaped response for tests and the dry run.

    ``match`` returns the same tokens for any prompt; ``diverge`` flips one
    token at position 40; ``offset`` shifts every logprob by -3e-3.
    """
    seed = int.from_bytes(hashlib.sha256(prompt_text.encode()).digest()[:4], "little")
    count = max_new_tokens
    token_ids = [(seed + i * 37) % 1000 for i in range(count)]
    logprobs = [-(0.05 + 0.001 * (i % 7)) for i in range(count)]
    top2 = [[[logprobs[i], token_ids[i]], [logprobs[i] - 0.2, (token_ids[i] + 1) % 1000]]
            for i in range(count)]
    if mode == "diverge" and count > 1:
        index = 12 if count > 12 else count - 1  # past the comparator's 8-token prefix
        token_ids[index] = (token_ids[index] + 1) % 1000
    elif mode == "offset":
        logprobs = [value - 3e-3 for value in logprobs]
        top2 = [[[logprobs[i], token_ids[i]], [logprobs[i] - 0.2, (token_ids[i] + 1) % 1000]]
                for i in range(count)]
    elif mode != "match":
        raise ValueError(f"unknown mode {mode!r}")
    return {
        "token_ids": token_ids,
        "logprobs": logprobs,
        "top2": top2,
        "completion_tokens": count,
        "accept_length": 3.2,
        "verify_ct": 40,
    }


class FakeHandler(BaseHTTPRequestHandler):
    mode = "match"
    seen: list = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") != "/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        FakeHandler.seen.append(payload)
        sampling = payload.get("sampling_params") or {}
        count = int(sampling.get("max_new_tokens", MAX_NEW_TOKENS))
        canned = canned_generate(payload.get("text", ""), count, FakeHandler.mode)
        response = {
            "text": "canned",
            "output_ids": canned["token_ids"],
            "meta_info": {
                "output_token_logprobs": [
                    [lp, tok, None] for lp, tok in zip(canned["logprobs"], canned["token_ids"])
                ],
                "output_top_logprobs": [
                    [[lp, tok, None] for lp, tok in position] for position in canned["top2"]
                ],
                "completion_tokens": canned["completion_tokens"],
                "spec_accept_length": canned["accept_length"],
                "spec_verify_ct": canned["verify_ct"],
            },
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the per-request line
        pass


def serve(port: int, mode: str) -> ThreadingHTTPServer:
    FakeHandler.mode = mode
    server = ThreadingHTTPServer(("127.0.0.1", port), FakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --- self test -------------------------------------------------------------

def _collect_args(tmp: Path, port: int, layout: str, name: str, mtp: bool) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=f"http://127.0.0.1:{port}",
        out=str(tmp / name),
        layout=layout,
        qpn2_tensors=None if layout == "marlin" else "both",
        commit="base" if layout == "marlin" else "cand",
        model="flash-next",
        checkpoint="ckpt",
        tp=4,
        mtp=mtp,
        tokenizer=None,
        reps=1,
        max_new_tokens=16,  # keep the fake runs quick; schema does not care
        timeout=10.0,
        dry_run=False,
        fake_mode="match",
    )


def selftest() -> int:
    here = Path(__file__).resolve().parent
    checks = 0
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)

        server = serve(0, "match")
        port = server.server_address[1]
        try:
            base = tmp / "base.json"
            args = _collect_args(tmp, port, "marlin", base.name, mtp=False)
            assert run_collect(args) == 0
            cand = tmp / "cand.json"
            args = _collect_args(tmp, port, "qpn2", cand.name, mtp=False)
            assert run_collect(args) == 0

            sent = FakeHandler.seen[0]
            assert sent.get("return_logprob") is True, sent
            assert sent.get("top_logprobs_num") == TOP_LOGGROBS, sent
            assert sent["sampling_params"]["temperature"] == 0.0, sent
            checks += 1
            print("ok  the client asks for logprobs, top-2 and greedy decoding")

            artifact = json.loads(cand.read_text())
            assert artifact["schema"] == SCHEMA
            prompt = artifact["prompts"][0]
            assert len(prompt["token_ids"]) == 16
            assert len(prompt["logprobs"]) == 16
            assert len(prompt["top2"]) == 16 and len(prompt["top2"][0]) == 2
            assert prompt["accept_length"] == 3.2
            assert artifact["launch"]["layout"] == "qpn2"
            checks += 1
            print("ok  the artifact is schema-shaped (ids, logprobs, top-2, accept)")

            comparator = here / "compare_runs.py"
            result = subprocess.run(
                [sys.executable, str(comparator), str(base), f"both={cand}"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            checks += 1
            print("ok  matching arms pass the comparator end to end")

            FakeHandler.mode = "offset"
            offset = tmp / "offset.json"
            args = _collect_args(tmp, port, "qpn2", offset.name, mtp=False)
            assert run_collect(args) == 0
            result = subprocess.run(
                [sys.executable, str(comparator), str(base), f"both={offset}"],
                capture_output=True, text=True,
            )
            assert result.returncode == 1, result.stdout + result.stderr
            assert "scale-path" in result.stdout, result.stdout
            checks += 1
            print("ok  an offset arm fails as scale-path end to end")

            FakeHandler.mode = "diverge"
            diverge = tmp / "diverge.json"
            args = _collect_args(tmp, port, "qpn2", diverge.name, mtp=False)
            assert run_collect(args) == 0
            result = subprocess.run(
                [sys.executable, str(comparator), str(base), f"both={diverge}"],
                capture_output=True, text=True,
            )
            assert result.returncode == 1 and "layout" in result.stdout, result.stdout
            checks += 1
            print("ok  a diverging arm fails with a layout cause end to end")

            # an unreachable server writes an artifact the comparator calls a crash
            dead = tmp / "dead.json"
            args = _collect_args(tmp, port, "qpn2", dead.name, mtp=False)
            args.base_url = "http://127.0.0.1:1"
            assert run_collect(args) == 1
            result = subprocess.run(
                [sys.executable, str(comparator), str(base), f"both={dead}"],
                capture_output=True, text=True,
            )
            assert result.returncode == 1 and "crash" in result.stdout, result.stdout
            checks += 1
            print("ok  an unreachable server writes an error artifact that fails as a crash")
        finally:
            server.shutdown()
            server.server_close()
            FakeHandler.seen.clear()

        # dry run without any server
        dry = tmp / "dry.json"
        args = argparse.Namespace(
            base_url="", out=str(dry), layout="qpn2", qpn2_tensors="both",
            commit="dry", model="m", checkpoint="c", tp=4, mtp=True,
            tokenizer=None, reps=2, max_new_tokens=MAX_NEW_TOKENS, timeout=5.0,
            dry_run=True, fake_mode="match",
        )
        assert run_collect(args) == 0
        artifact = json.loads(dry.read_text())
        assert len(artifact["prompts"]) == len(CONTEXTS) * 2 * 2, len(artifact["prompts"])
        assert artifact["launch"]["mtp"] is True
        checks += 1
        print("ok  --dry-run writes a 16-prompt artifact with no server")

    print(f"selftest: {checks} checks passed")
    return 0


# --- CLI -------------------------------------------------------------------

def _tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="drive a server and write one artifact")
    collect.add_argument("--base-url", default="http://127.0.0.1:30000")
    collect.add_argument("--out", required=True)
    collect.add_argument("--layout", choices=("marlin", "qpn2"), required=True)
    collect.add_argument("--qpn2-tensors", choices=("both", "w13", "w2"))
    collect.add_argument("--commit", required=True)
    collect.add_argument("--model", default="Qwen3.8-Flash-Next")
    collect.add_argument("--checkpoint", default="")
    collect.add_argument("--tp", type=int, default=4)
    collect.add_argument("--mtp", action="store_true")
    collect.add_argument("--tokenizer", help="tokenizer path; omit only with --dry-run")
    collect.add_argument("--reps", type=int, default=REPS)
    collect.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    collect.add_argument("--timeout", type=float, default=1800.0)
    collect.add_argument("--dry-run", action="store_true",
                         help="write the artifact from canned responses, no server")
    collect.add_argument("--fake-mode", choices=("match", "diverge", "offset"), default="match")

    fake = sub.add_parser("fake-server", help="serve canned /generate responses")
    fake.add_argument("--port", type=int, default=18400)
    fake.add_argument("--mode", choices=("match", "diverge", "offset"), default="match")

    sub.add_parser("selftest", help="fake server + both arms + the comparator")

    args = parser.parse_args()
    if args.command == "fake-server":
        server = serve(args.port, args.mode)
        print(f"fake /generate on http://127.0.0.1:{server.server_address[1]} mode={args.mode}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.shutdown()
            return 0
    if args.command == "selftest":
        return selftest()

    if not args.dry_run and not args.tokenizer:
        parser.error("--tokenizer is required unless --dry-run is set")
    if args.qpn2_tensors and args.layout != "qpn2":
        parser.error("--qpn2-tensors only applies to --layout qpn2")
    args.tokenizer = _tokenizer(args.tokenizer) if not args.dry_run else None
    return run_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
