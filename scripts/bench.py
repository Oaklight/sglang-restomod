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


def spec_fields(meta):
    """Accept metrics from a final ``meta_info``, or {} when spec decoding is off.

    The engine fills these only when it verified speculative drafts, so their
    absence is how a non-spec launch is told apart from a spec launch that
    accepted nothing.
    """
    if not meta:
        return {}
    out = {}
    for key in ("spec_accept_length", "spec_accept_rate"):
        if meta.get(key) is not None:
            out[key.replace("spec_", "")] = round(float(meta[key]), 3)
    if meta.get("spec_verify_ct"):
        out["verify_ct"] = int(meta["spec_verify_ct"])
    return out


def stream_request(url, payload, timeout):
    """Return timing, usage and the final ``meta_info`` of one streamed request."""
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    first_t = None
    n_tok = 0
    chunk_t = []
    usage = None
    meta = None
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
            if obj.get("meta_info"):
                meta = obj["meta_info"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            # chat completions stream deltas; legacy completions stream `text`.
            delta = choice.get("delta") or {}
            txt = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
            if txt:
                now = time.perf_counter()
                if first_t is None:
                    first_t = now
                n_tok += 1
                chunk_t.append(now)
    end_t = time.perf_counter()
    return {"t0": t0, "first_t": first_t, "end_t": end_t, "n_tok": n_tok,
            "chunk_t": chunk_t, "usage": usage, "meta": meta}


def probe_spec_accept(host, prompt, output_tokens, timeout):
    """One short ``/generate`` request, for the accept metrics it returns.

    The OpenAI-compatible streaming path carries ``usage`` but not
    ``meta_info``, so the timed reps cannot see the accept rate. ``/generate``
    does return it, and accept length is a property of the draft rather than of
    the request, so one probe per context is enough. The probe asks for the
    bench's own output length, capped at 64: on a shorter reply the accept
    length is dominated by the first few verify passes.
    """
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": min(output_tokens, 64),
                            "temperature": 0.0, "ignore_eos": True},
        "stream": True,
    }
    r = stream_request(f"http://{host}/generate", payload, timeout)
    return spec_fields(r.get("meta"))


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
        **spec_fields(r.get("meta")),
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
    ap.add_argument("--no-spec-accept", dest="spec_accept", action="store_false",
                    help="skip the /generate probe that records the accept "
                         "length and rate (run when the endpoint has no /generate)")
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
                # n_tok counts streamed chunks, and MTP puts several accepted
                # tokens in one chunk, so it undercounts; use the usage block.
                # The window starts at the earliest first token and ends at the
                # latest final token, which keeps prefill out of the rate.
                # Two windows, because they answer different questions.
                # burst: earliest first token to latest last token. It is what
                # a cold batch costs end to end, and it carries the stagger of
                # the later requests' prefill.
                # active: latest first token to earliest last token. Every
                # request is decoding inside it, so its rate is what the server
                # sustains with the batch full.
                # Tokens come from the usage block, since MTP packs several
                # accepted tokens into one chunk and the chunk count undercounts.
                # They cannot be timestamped, so the active window counts chunks
                # and scales each request by its own tokens-per-chunk ratio.
                done = [o for o in out if o and o["first_t"] is not None]
                tot = 0
                burst_s = 0.0
                active_s = 0.0
                active_tot = 0.0
                if done:
                    for o in done:
                        o["tokens"] = max(
                            ((o["usage"] or {}).get("completion_tokens")
                             or o["n_tok"]) - 1,
                            0,
                        )
                    tot = sum(o["tokens"] for o in done)
                    burst_s = max(o["end_t"] for o in done) - min(
                        o["first_t"] for o in done
                    )
                    b_start = max(o["first_t"] for o in done)
                    b_end = min(o["end_t"] for o in done)
                    active_s = b_end - b_start
                    if active_s > 0:
                        for o in done:
                            inside = sum(1 for t in o["chunk_t"] if b_start <= t <= b_end)
                            if o["n_tok"]:
                                active_tot += o["tokens"] * inside / o["n_tok"]
                    else:
                        active_s = 0.0
                burst_s = max(burst_s, 1e-6)
                # Accept length is tokens per verify pass, so it pools by
                # verify count, not by request: the batch shares no state.
                verify = sum(
                    (o["meta"] or {}).get("spec_verify_ct") or 0 for o in done
                )
                accept = round(tot / verify, 3) if verify else None
                s = {"label": f"ctx{ctx} C{args.concurrency} rep{rep+1}",
                     "requests": args.concurrency,
                     "tokens": tot,
                     "accept_length": accept,
                     "aggregate_decode_tps": (
                         round(active_tot / active_s, 2) if active_s > 0 else None
                     ),
                     "per_request_tps": (
                         round(active_tot / active_s / max(len(done), 1), 2)
                         if active_s > 0 else None
                     ),
                     "active_s": round(active_s, 3),
                     "burst_decode_tps": round(tot / burst_s, 2),
                     "burst_s": round(burst_s, 3),
                     "wall_s": round(wall, 3)}
                results.append(s)
                print(json.dumps(s))

        # After the timed reps, so the probe cannot move a number. A server
        # without speculative decoding reports no fields; that is recorded, not
        # treated as a failure.
        if args.spec_accept:
            try:
                fields = probe_spec_accept(
                    args.host, prompt, args.output_tokens, args.timeout
                )
                row = {"label": f"ctx{ctx} spec-accept", **fields}
                if not fields:
                    row["note"] = "no spec metrics in the response"
            except Exception as exc:  # noqa: BLE001 - a probe must not fail the bench
                row = {"label": f"ctx{ctx} spec-accept", "error": str(exc)}
            results.append(row)
            print(json.dumps(row))

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
