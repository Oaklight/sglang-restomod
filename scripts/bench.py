#!/usr/bin/env python3
"""Decode/prefill benchmark for an OpenAI-compatible endpoint.

Methodology follows the 1Cat-LLM release contract so numbers are comparable:
  pure decode = (output_tokens - 1) / (last_chunk_t - first_chunk_t)
i.e. TTFT/prefill is excluded from the decode rate.

Prompts are built with the model's own tokenizer at exact token counts.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

MODEL_DIR = os.environ.get("QLAB_MODEL_DIR", "")
API_KEY = os.environ.get("VLLM_API_KEY", "")



def build_prompt(tok, n_tokens, seed_text):
    """Return a prompt whose tokenized length is as close to n_tokens as possible."""
    ids = tok.encode(seed_text, add_special_tokens=False)
    if not ids:
        raise SystemExit("seed text produced no tokens")
    reps = (n_tokens + len(ids) - 1) // len(ids)
    full = (ids * reps)[:n_tokens]
    return tok.decode(full), len(full)


def stream_request(url, payload, timeout):
    """Yield (kind, value) where kind is 'first' | 'token' | 'usage'."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    first_t = None
    n_tok = 0
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            # chat completions stream deltas; legacy completions stream `text`.
            delta = choice.get("delta") or {}
            txt = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
            if txt:
                if first_t is None:
                    first_t = time.perf_counter()
                n_tok += 1
    end_t = time.perf_counter()
    return {"t0": t0, "first_t": first_t, "end_t": end_t, "n_tok": n_tok, "usage": usage}


def summarize(r, label):
    if r["first_t"] is None or r["n_tok"] == 0:
        return {"label": label, "error": "no output tokens"}
    ttft = r["first_t"] - r["t0"]
    dec = r["end_t"] - r["first_t"]
    prompt_tok = (r["usage"] or {}).get("prompt_tokens")
    out_tok = (r["usage"] or {}).get("completion_tokens") or r["n_tok"]
    return {
        "label": label,
        "prompt_tokens": prompt_tok,
        "output_tokens": out_tok,
        "ttft_s": round(ttft, 3),
        "decode_s": round(dec, 3),
        "pure_decode_tps": round(max(out_tok - 1, 0) / dec, 2) if dec > 0 else None,
        "prefill_tps": round(prompt_tok / ttft, 1) if (prompt_tok and ttft > 0) else None,
        "total_s": round(r["end_t"] - r["t0"], 3),
    }


def main():
    global API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b-dflash2")
    ap.add_argument("--contexts", default="4096,32768")
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-eos", action="store_true", default=True)
    ap.add_argument("--chat", action="store_true",
                    help="use /v1/chat/completions instead of /v1/completions")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--fresh", action="store_true",
                    help="give each context a unique prefix so prefill is cold")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="tokenizer source; defaults to $QLAB_MODEL_DIR")
    ap.add_argument("--api-key", default=API_KEY,
                    help="bearer token; defaults to $VLLM_API_KEY")
    args = ap.parse_args()

    API_KEY = args.api_key
    if not args.model_dir:
        raise SystemExit("--model-dir or QLAB_MODEL_DIR is required for the tokenizer")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    seed_text = (
        "The following is a technical report about distributed inference systems. "
        "Memory bandwidth, kernel occupancy, and interconnect topology determine "
        "the achievable token rate for autoregressive decoding. "
    )
    url = f"http://{args.host}/v1/" + ("chat/completions" if args.chat else "completions")
    results = []

    for ctx in [int(x) for x in args.contexts.split(",")]:
        salt = f"[run {os.urandom(4).hex()} ctx {ctx}] " if args.fresh else ""
        prompt, plen = build_prompt(tok, ctx, salt + seed_text)
        payload = {
            "model": args.model,
            "max_tokens": args.output_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "seed": 0,
        }
        if args.chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["prompt"] = prompt
        def refresh_prompt():
            """Each fresh request needs its own salt at the START of the prompt,
            otherwise prefix caching turns every rep into a cache hit."""
            if not args.fresh:
                return
            p, _ = build_prompt(tok, ctx, f"[run {os.urandom(4).hex()} ctx {ctx}] " + seed_text)
            if args.chat:
                payload["messages"] = [{"role": "user", "content": p}]
            else:
                payload["prompt"] = p

        for w in range(args.warmups):
            refresh_prompt()
            stream_request(url, payload, args.timeout)
            print(f"  [warmup {w+1}] ctx={ctx}", file=sys.stderr, flush=True)

        for rep in range(args.reps):
            refresh_prompt()
            if args.concurrency == 1:
                r = stream_request(url, payload, args.timeout)
                s = summarize(r, f"ctx{ctx} rep{rep+1}")
                results.append(s)
                print(json.dumps(s))
            else:
                out = [None] * args.concurrency
                def work(i):
                    out[i] = stream_request(url, payload, args.timeout)
                ts = [threading.Thread(target=work, args=(i,)) for i in range(args.concurrency)]
                t0 = time.perf_counter()
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                wall = time.perf_counter() - t0
                tot = sum(o["n_tok"] for o in out if o)
                agg = tot / wall
                s = {"label": f"ctx{ctx} C{args.concurrency} rep{rep+1}",
                     "requests": args.concurrency,
                     "aggregate_decode_tps": round(agg, 2),
                     "wall_s": round(wall, 3)}
                results.append(s)
                print(json.dumps(s))

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
