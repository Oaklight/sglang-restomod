#!/usr/bin/env python3
"""Cross-run comparator for the QPN2 P3/P4 guard (#224).

Takes the artifacts of a baseline run (Marlin everywhere) and one or more
candidate runs (QPN2 everywhere, or one tensor at a time) and applies the
pass/fail rules and the localization table in ``docs/qpn2-functional-guard.md``
section 4. It never touches a GPU and never reads a checkpoint: everything it
judges is in the artifacts.

Rules, in the doc's terms:

- pass: identical 128-token greedy sequences on every prompt and rep, and
  ``max |delta logprob| <= 1e-3`` nats on the positions where the tokens agree;
- near-tie pass: a token mismatch is exempt when the baseline top-2 gap is
  ``< 1e-3`` and the candidate chose the baseline's second token, and only from
  token 8 onward;
- fail: any other mismatch, any logprob drift beyond 1e-3, a truncated or
  missing prompt, a launch error, or an accept-length drop above 0.05 with
  matching tokens (the draft path).

Artifact schema (JSON, one file per run)::

    {
      "schema": "qpn2-run-v1",
      "launch": {
        "commit": "...", "layout": "marlin" | "qpn2",
        "qpn2_tensors": "both" | "w13" | "w2", "mtp": false,
        "tp": 4, "model": "...", "checkpoint": "...",
        "flags": {"--mem-fraction-static": "0.76"}
      },
      "error": null,
      "prompts": [
        {"id": "ctx76-a-r0", "context_tokens": 76, "prompt_tokens": 76,
         "token_ids": [128 ints],
         "logprobs": [128 floats, the chosen token's logprob],
         "top2": [[[lp1, tok1], [lp2, tok2]], ...],   # optional, needed for near ties
         "accept_length": 3.21, "verify_ct": 100, "completion_tokens": 320}
      ]
    }

Usage::

    python compare_runs.py BASELINE.json CANDIDATE [CANDIDATE ...]
    python compare_runs.py --selftest

A candidate is ``PATH`` or ``LABEL=PATH``; the label is used in the report and
to pair the per-tensor arms (``w13=...``, ``w2=...``). Exit codes: 0 all pass,
1 a verdict failed, 2 the inputs could not be judged (incomparable runs, MTP on,
malformed artifact).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCHEMA = "qpn2-run-v1"
TOL = 1e-3
NEAR_TIE = 1e-3
MIN_PREFIX = 8
ACCEPT_TOL = 0.05
GROWTH_RATIO = 3.0

PASS = "pass"
PASS_NEAR = "pass (near-tie)"
FAIL = "fail"
NOT_COMPARED = "not compared"


# --- artifact loading ------------------------------------------------------

class UsageError(Exception):
    """The inputs cannot be judged; exit 2."""


def load_run(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise UsageError(f"{path}: not found") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path}: not JSON ({exc})") from exc
    if data.get("schema") != SCHEMA:
        raise UsageError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    launch = data.get("launch")
    if not isinstance(launch, dict):
        raise UsageError(f"{path}: missing launch metadata")
    for key in ("layout", "commit", "mtp"):
        if key not in launch:
            raise UsageError(f"{path}: launch.{key} is required")
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise UsageError(f"{path}: no prompts")
    seen = set()
    for prompt in prompts:
        for key in ("id", "token_ids", "logprobs"):
            if key not in prompt:
                raise UsageError(f"{path}: prompt {prompt.get('id')!r} lacks {key}")
        if len(prompt["token_ids"]) != len(prompt["logprobs"]):
            raise UsageError(f"{path}: prompt {prompt['id']!r} token/logprob lengths differ")
        if prompt["id"] in seen:
            raise UsageError(f"{path}: duplicate prompt id {prompt['id']!r}")
        seen.add(prompt["id"])
    return data


def comparable(baseline: dict, candidate: dict) -> str | None:
    """Why the two runs cannot be compared, or None."""
    for key in ("model", "checkpoint"):
        if str(baseline["launch"].get(key)) != str(candidate["launch"].get(key)):
            return (
                f"launch.{key} differs ({baseline['launch'].get(key)!r} vs "
                f"{candidate['launch'].get(key)!r})"
            )
    if int(baseline["launch"].get("tp", 0)) != int(candidate["launch"].get("tp", 0)):
        return "launch.tp differs"
    return None


# --- per-prompt comparison -------------------------------------------------

def _top2(prompt: dict, index: int):
    top2 = prompt.get("top2")
    if not top2 or index >= len(top2):
        return None
    pair = top2[index]
    if not pair or len(pair) < 2 or len(pair[0]) < 2 or len(pair[1]) < 2:
        return None
    return pair[0], pair[1]


def near_tie_exempt(
    baseline_prompt: dict, index: int, candidate_token: int
) -> tuple[bool, float | None]:
    top = _top2(baseline_prompt, index)
    if top is None:
        return False, None
    (first_lp, _), (second_lp, second_tok) = top
    gap = first_lp - second_lp
    return (gap < NEAR_TIE and candidate_token == second_tok and index >= MIN_PREFIX), gap


def _drift_profile(deltas: list[float]) -> dict:
    """Mean/std of the signed delta, and the first/last quarter magnitudes."""
    if not deltas:
        return {"mean": 0.0, "std": 0.0, "first": 0.0, "last": 0.0}
    quarter = max(1, len(deltas) // 4)
    first = statistics.fmean(abs(d) for d in deltas[:quarter])
    last = statistics.fmean(abs(d) for d in deltas[-quarter:])
    return {
        "mean": statistics.fmean(deltas),
        "std": statistics.pstdev(deltas),
        "first": first,
        "last": last,
    }


def _is_scale_path(deltas: list[float], base_lps: list[float]) -> bool:
    """Constant offset, or a drift proportional to the logprob magnitude."""
    profile = _drift_profile(deltas)
    if abs(profile["mean"]) > TOL and profile["std"] < 0.1 * abs(profile["mean"]):
        return True
    if len(set(round(x, 9) for x in base_lps)) > 3:
        pairs = [(x, d) for x, d in zip(base_lps, deltas)]
        mean_x = statistics.fmean(x for x, _ in pairs)
        mean_d = statistics.fmean(d for _, d in pairs)
        var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
        if var_x > 0:
            slope = sum((x - mean_x) * (d - mean_d) for x, d in pairs) / var_x
            residual = [d - (mean_d + slope * (x - mean_x)) for x, d in pairs]
            spread = statistics.pstdev(deltas)
            if abs(slope) > TOL and spread > 0 and statistics.pstdev(residual) < 0.25 * spread:
                return True
    return False


def compare_prompt(baseline_prompt: dict, candidate_prompt: dict) -> dict:
    base_tokens = baseline_prompt["token_ids"]
    cand_tokens = candidate_prompt["token_ids"]
    base_lps = baseline_prompt["logprobs"]
    cand_lps = candidate_prompt["logprobs"]

    result = {
        "id": baseline_prompt["id"],
        "context_tokens": baseline_prompt.get("context_tokens"),
        "status": PASS,
        "cause": None,
        "note": "",
        "first_mismatch": None,
        "mismatches": [],
        "near_ties": [],
        "max_dlogp": 0.0,
        "accept_base": baseline_prompt.get("accept_length"),
        "accept_cand": candidate_prompt.get("accept_length"),
    }

    if len(cand_tokens) < len(base_tokens):
        result.update(status=FAIL, cause="truncation",
                      note=f"{len(cand_tokens)} of {len(base_tokens)} tokens")
        return result

    equal_positions = []
    for index in range(min(len(base_tokens), len(cand_tokens))):
        if base_tokens[index] != cand_tokens[index]:
            exempt, gap = near_tie_exempt(baseline_prompt, index, cand_tokens[index])
            if exempt:
                result["near_ties"].append((index, gap))
            else:
                result["mismatches"].append(index)
                if result["first_mismatch"] is None:
                    result["first_mismatch"] = index
        else:
            equal_positions.append(index)

    deltas = [cand_lps[i] - base_lps[i] for i in equal_positions]
    profile = _drift_profile(deltas)
    result["max_dlogp"] = max((abs(d) for d in deltas), default=0.0)

    if result["mismatches"]:
        first = result["first_mismatch"]
        result["status"] = FAIL
        result["cause"] = "first-call" if first < MIN_PREFIX else "layout"
        result["note"] = (
            f"{len(result['mismatches'])} mismatch(es), first at token {first}"
            + (f", {len(result['near_ties'])} near-tie exempt" if result["near_ties"] else "")
        )
    elif result["max_dlogp"] > TOL:
        result["status"] = FAIL
        if _is_scale_path(deltas, [base_lps[i] for i in equal_positions]):
            result["cause"] = "scale-path"
        elif profile["first"] > 0 and profile["last"] > GROWTH_RATIO * profile["first"]:
            result["cause"] = "accumulation"
        else:
            result["cause"] = "logprob-drift"
        result["note"] = (
            f"max|dlogp|={result['max_dlogp']:.2e}, mean={profile['mean']:+.2e}, "
            f"std={profile['std']:.2e}, q1={profile['first']:.2e} q4={profile['last']:.2e}"
        )
    elif result["near_ties"]:
        result["status"] = PASS_NEAR
        result["note"] = f"{len(result['near_ties'])} near-tie mismatch(es)"

    return result


# --- run-level comparison --------------------------------------------------

def compare_runs(baseline: dict, candidate: dict, mtp_secondary: bool) -> dict:
    base_by_id = {p["id"]: p for p in baseline["prompts"]}
    cand_by_id = {p["id"]: p for p in candidate["prompts"]}

    report = {
        "candidate": candidate["launch"],
        "status": PASS,
        "cause": None,
        "notes": [],
        "prompts": [],
        "max_dlogp": 0.0,
        "accept_base": None,
        "accept_cand": None,
    }

    if candidate.get("error"):
        report.update(status=FAIL, cause="crash",
                      notes=[str(candidate["error"]).splitlines()[0][:200]])
        report["prompts"] = [
            {"id": p["id"], "status": FAIL, "cause": "crash", "note": "run errored",
             "mismatches": [], "near_ties": [], "max_dlogp": 0.0}
            for p in baseline["prompts"]
        ]
        return report

    for pid, base_prompt in base_by_id.items():
        cand_prompt = cand_by_id.get(pid)
        if cand_prompt is None:
            report["prompts"].append(
                {"id": pid, "context_tokens": base_prompt.get("context_tokens"),
                 "status": FAIL, "cause": "missing-prompt", "note": "candidate has no such prompt",
                 "mismatches": [], "near_ties": [], "max_dlogp": 0.0}
            )
            continue
        report["prompts"].append(compare_prompt(base_prompt, cand_prompt))

    extra = set(cand_by_id) - set(base_by_id)
    if extra:
        sample = sorted(extra)[:3]
        report["notes"].append(
            f"candidate has {len(extra)} prompt(s) the baseline lacks: {sample}"
        )

    base_accept = [
        p.get("accept_length") for p in baseline["prompts"] if p.get("accept_length") is not None
    ]
    cand_accept = [
        cand_by_id[p["id"]].get("accept_length")
        for p in baseline["prompts"]
        if p["id"] in cand_by_id and cand_by_id[p["id"]].get("accept_length") is not None
    ]
    if base_accept and cand_accept:
        report["accept_base"] = statistics.fmean(base_accept)
        report["accept_cand"] = statistics.fmean(cand_accept)

    failures = [p for p in report["prompts"] if p["status"] == FAIL]
    report["max_dlogp"] = max((p["max_dlogp"] for p in report["prompts"]), default=0.0)

    if mtp_secondary:
        # Token equality is not attributable while rejection sampling is on;
        # judge the accept length and logprob drift only.
        for prompt in report["prompts"]:
            prompt["token_status"] = prompt["status"]
            prompt["status"] = (
                FAIL if prompt["cause"] in ("scale-path", "accumulation", "logprob-drift",
                                            "truncation", "missing-prompt", "crash")
                else NOT_COMPARED
            )
        failures = [p for p in report["prompts"] if p["status"] == FAIL]
        if report["accept_base"] is not None and report["accept_cand"] is not None:
            drop = report["accept_base"] - report["accept_cand"]
            if drop > ACCEPT_TOL:
                report.update(status=FAIL, cause="draft")
                report["notes"].append(
                    f"accept length {report['accept_base']:.2f} -> {report['accept_cand']:.2f}"
                )
            elif failures:
                report.update(status=FAIL, cause=failures[0]["cause"])
            elif report["max_dlogp"] > TOL:
                report.update(status=FAIL, cause="logprob-drift")
            else:
                report.update(status=PASS)
            return report

    if not failures:
        report["status"] = PASS
        if report["accept_base"] is not None and report["accept_cand"] is not None:
            drop = report["accept_base"] - report["accept_cand"]
            if drop > ACCEPT_TOL:
                report.update(status=FAIL, cause="draft",
                              notes=report["notes"] + [
                                  f"accept length {report['accept_base']:.2f} -> "
                                  f"{report['accept_cand']:.2f} while the tokens match"]
                              )
        if report["status"] == PASS and any(p["status"] == PASS_NEAR for p in report["prompts"]):
            report["status"] = PASS_NEAR
            report["notes"].append("mismatches are all near ties")
        return report

    report["status"] = FAIL
    report["cause"] = failures[0]["cause"]

    # Divergence confined to the longest context is the address/last-tile case.
    contexts = {
        p["context_tokens"] for p in report["prompts"] if p.get("context_tokens") is not None
    }
    if len(contexts) > 1:
        failing = {p["context_tokens"] for p in failures}
        passing = {
            p["context_tokens"] for p in report["prompts"]
            if p["status"] != FAIL and p.get("context_tokens") is not None
        }
        if failing and failing == {max(contexts)} and passing and max(contexts) not in passing:
            report["cause"] = "long-context"
    return report


def apply_arm_mapping(reports: dict[str, dict]) -> None:
    """Turn an unlocalised layout failure into w13 or w2 when the arms say so."""
    def tensors(report):
        return report["candidate"].get("qpn2_tensors")

    for label, report in reports.items():
        if report["status"] != FAIL or report["cause"] != "layout":
            continue
        mine = tensors(report)
        if mine not in ("w13", "w2"):
            continue
        other = "w2" if mine == "w13" else "w13"
        siblings = [r for l, r in reports.items() if l != label and tensors(r) == other]
        if siblings and all(r["status"] != FAIL for r in siblings):
            report["cause"] = f"{mine}-layout"


# --- reporting -------------------------------------------------------------

CAUSE_TEXT = {
    "first-call": "first decode call: expert addressing or the expert stride",
    "layout": "layout permutation (arm not isolated)",
    "w13-layout": "w13 layout: tile/lane mapping, N=320 (10 tiles)",
    "w2-layout": "w2 layout: group split/stride, K=160 (10 groups)",
    "scale-path": "scale path: fp16 RNE vs float32, or the global-scale multiply",
    "accumulation": "drift grows with position: accumulation or a per-layer stride",
    "logprob-drift": "logprob drift without a token change",
    "long-context": "only the longest context fails: address rule, padding or last tile",
    "draft": "draft path: accept length dropped while the target tokens match",
    "truncation": "candidate stopped early",
    "missing-prompt": "candidate lacks a prompt",
    "crash": "the candidate run errored",
}


def describe(report: dict) -> str:
    lines = []
    cause = report["cause"]
    if cause:
        lines.append(f"  cause: {cause} - {CAUSE_TEXT.get(cause, cause)}")
    for prompt in report["prompts"]:
        lines.append(
            f"  {prompt['id']:<28} {prompt['status']:<16} "
            f"max|dlogp|={prompt['max_dlogp']:.2e}"
            + (f"  {prompt['note']}" if prompt.get("note") else "")
        )
    for note in report["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def run(paths: dict[str, str], mtp_secondary: bool) -> tuple[int, dict]:
    baseline = load_run(paths["baseline"])
    for label, path in paths["candidates"]:
        if path == paths["baseline"]:
            raise UsageError(f"candidate {label} is the baseline file")
    candidates = {label: load_run(path) for label, path in paths["candidates"]}

    if baseline["launch"].get("mtp") and not mtp_secondary:
        raise UsageError(
            "the baseline ran with MTP on; rejection sampling is stochastic even at "
            "temperature 0, so token equality is not attributable. Run the primary "
            "arm with MTP off, or pass --mtp-secondary to judge accept length and "
            "logprob drift only."
        )

    reports = {}
    for label, candidate in candidates.items():
        if candidate["launch"].get("mtp") and not mtp_secondary:
            raise UsageError(f"candidate {label} ran with MTP on; see --mtp-secondary")
        reason = comparable(baseline, candidate)
        if reason:
            raise UsageError(f"candidate {label} is not comparable: {reason}")
        reports[label] = compare_runs(baseline, candidate, mtp_secondary)

    apply_arm_mapping(reports)

    payload = {
        "baseline": baseline["launch"],
        "candidates": {label: report for label, report in reports.items()},
        "verdict": FAIL if any(r["status"] == FAIL for r in reports.values()) else PASS,
    }
    return (0 if payload["verdict"] == PASS else 1), payload


def print_report(paths: dict[str, str], payload: dict) -> None:
    base = payload["baseline"]
    print(
        f"baseline  {paths['baseline']}  layout={base.get('layout')} "
        f"tensors={base.get('qpn2_tensors')} mtp={base.get('mtp')} commit={base.get('commit')}"
    )
    for label, report in payload["candidates"].items():
        launch = report["candidate"]
        print(
            f"candidate {label}: layout={launch.get('layout')} "
            f"tensors={launch.get('qpn2_tensors')} mtp={launch.get('mtp')} "
            f"commit={launch.get('commit')}  ->  {report['status'].upper()}"
        )
        print(describe(report))
    print(f"verdict: {payload['verdict'].upper()}")


# --- self test -------------------------------------------------------------

def _lp(i: int) -> float:
    return -0.05 - 0.001 * (i % 7)


def _prompt(pid: str, context: int, tokens=None, lps=None, top2=None, accept=None) -> dict:
    tokens = tokens or [100 + (i * 7) % 50 for i in range(128)]
    prompt = {
        "id": pid,
        "context_tokens": context,
        "prompt_tokens": context,
        "token_ids": tokens,
        "logprobs": lps or [_lp(i) for i in range(len(tokens))],
        "completion_tokens": len(tokens),
    }
    if accept is not None:
        prompt["accept_length"] = accept
        prompt["verify_ct"] = 100
    if top2 is not None:
        prompt["top2"] = top2
    return prompt


def _run(
    layout="marlin", tensors=None, mtp=False, prompts=None, error=None, checkpoint="ckpt"
) -> dict:
    return {
        "schema": SCHEMA,
        "launch": {
            "commit": "base" if layout == "marlin" else "cand",
            "layout": layout,
            "qpn2_tensors": tensors,
            "mtp": mtp,
            "tp": 4,
            "model": "flash-next",
            "checkpoint": checkpoint,
            "flags": {},
        },
        "error": error,
        "prompts": prompts or [_prompt("ctx76-a-r0", 76), _prompt("ctx7168-a-r0", 7168)],
    }


def _mutate(run: dict, fn) -> dict:
    out = json.loads(json.dumps(run))
    for prompt in out["prompts"]:
        fn(prompt)
    return out


def _selftest() -> int:
    checks = 0

    def expect(name, baseline, candidates, want_status, want_cause=None, mtp_secondary=False):
        nonlocal checks
        reports = {}
        for label, cand in candidates.items():
            reason = comparable(baseline, cand)
            assert reason is None, f"{name}: {reason}"
            reports[label] = compare_runs(baseline, cand, mtp_secondary)
        apply_arm_mapping(reports)
        for label, report in reports.items():
            assert report["status"] == want_status, (
                f"{name}/{label}: status {report['status']}, want {want_status}"
                + "\n"
                + describe(report)
            )
            if want_cause is not None:
                assert report["cause"] == want_cause, (
                    f"{name}/{label}: cause {report['cause']}, want {want_cause}"
                    + "\n"
                    + describe(report)
                )
        checks += 1
        print(f"ok  {name}")

    base = _run()

    # 1. identical
    expect("identical runs pass", base, {"both": _run("qpn2", "both")}, PASS)

    # 2. mismatch inside the first 8 tokens
    def early(p):
        p["token_ids"][3] = 7
    expect("mismatch at token 3 is a first-call failure", base,
           {"both": _mutate(_run("qpn2", "both"), early)}, FAIL, "first-call")

    # 3. near tie after the prefix. The gap is the baseline's, so the top-2 is
    #    attached to the baseline artifact.
    def late(p):
        p["token_ids"][40] = 201

    base_top2 = json.loads(json.dumps(base))
    for prompt in base_top2["prompts"]:
        prompt["top2"] = [[[-0.1, tok], [-0.2, 999]] for tok in prompt["token_ids"]]
        prompt["top2"][40] = [[-0.1000, prompt["token_ids"][40]], [-0.1005, 201]]
    near = _mutate(_run("qpn2", "both"), late)
    expect("near tie after token 8 passes", base_top2, {"both": near}, PASS_NEAR)

    # 4. the same mismatch without a near tie: no top-2, so no exemption
    expect("mismatch after token 8 without a near tie fails", base,
           {"both": _mutate(_run("qpn2", "both"), late)}, FAIL, "layout")

    # 5. near tie inside the first 8 tokens is not exempt
    base_top2_early = json.loads(json.dumps(base))
    for prompt in base_top2_early["prompts"]:
        prompt["top2"] = [[[-0.1, tok], [-0.2, 999]] for tok in prompt["token_ids"]]
        prompt["top2"][3] = [[-0.1000, prompt["token_ids"][3]], [-0.1005, 7]]
    expect("near tie at token 3 is still a failure", base_top2_early,
           {"both": _mutate(_run("qpn2", "both"), early)}, FAIL, "first-call")

    # 6. constant logprob offset
    def offset(p):
        p["logprobs"] = [x - 3e-3 for x in p["logprobs"]]
    expect("constant offset maps to the scale path", base,
           {"both": _mutate(_run("qpn2", "both"), offset)}, FAIL, "scale-path")

    # 7. proportional drift
    def proportional(p):
        p["logprobs"] = [x * 1.05 for x in p["logprobs"]]
    expect("proportional drift maps to the scale path", base,
           {"both": _mutate(_run("qpn2", "both"), proportional)}, FAIL, "scale-path")

    # 8. drift that grows with position
    def growing(p):
        n = len(p["logprobs"])
        p["logprobs"] = [x - 2e-2 * i / n for i, x in enumerate(p["logprobs"])]
    expect("growing drift maps to accumulation", base,
           {"both": _mutate(_run("qpn2", "both"), growing)}, FAIL, "accumulation")

    # 9. only the longest prompt fails
    long_base = _run(prompts=[_prompt("ctx76", 76), _prompt("ctx7168", 7168)])
    long_cand = json.loads(json.dumps(long_base))
    long_cand["launch"].update(layout="qpn2", qpn2_tensors="both", commit="cand")
    for prompt in long_cand["prompts"]:
        if prompt["context_tokens"] == 7168:
            prompt["token_ids"][100] = 9
    expect("only the longest context failing maps to long-context", long_base,
           {"both": long_cand}, FAIL, "long-context")

    # 10. per-tensor arms
    reports = {}
    for label, cand in {
        "both": _run("qpn2", "both"),
        "w13": _mutate(_run("qpn2", "w13"), late),
        "w2": _run("qpn2", "w2"),
    }.items():
        reports[label] = compare_runs(base, cand, False)
    apply_arm_mapping(reports)
    assert reports["both"]["status"] == PASS
    assert reports["w13"]["cause"] == "w13-layout", reports["w13"]["cause"]
    checks += 1
    print("ok  w13-only failure with w2 passing maps to the w13 layout")

    reports = {}
    for label, cand in {
        "both": _run("qpn2", "both"),
        "w13": _run("qpn2", "w13"),
        "w2": _mutate(_run("qpn2", "w2"), late),
    }.items():
        reports[label] = compare_runs(base, cand, False)
    apply_arm_mapping(reports)
    assert reports["w13"]["status"] == PASS
    assert reports["w2"]["cause"] == "w2-layout", reports["w2"]["cause"]
    checks += 1
    print("ok  w2-only failure with w13 passing maps to the w2 layout")

    # 11. draft: accept drop with matching tokens
    draft_base = _run(prompts=[_prompt("p", 76, accept=3.4)])
    draft_cand = _run("qpn2", "both", prompts=[_prompt("p", 76, accept=3.2)])
    expect("accept drop with matching tokens maps to the draft", draft_base,
           {"both": draft_cand}, FAIL, "draft")

    # 12. truncation
    short = _mutate(_run("qpn2", "both"), lambda p: p.update(
        token_ids=p["token_ids"][:100], logprobs=p["logprobs"][:100]))
    expect("truncated candidate fails as truncation", base, {"both": short}, FAIL, "truncation")

    # 13. crash
    crashed = _run("qpn2", "both", error="RuntimeError: kernel launch failed")
    expect("errored run fails as a crash", base, {"both": crashed}, FAIL, "crash")

    # 14. incomparable
    other = _run("qpn2", "both", checkpoint="other-ckpt")
    assert comparable(base, other) is not None
    checks += 1
    print("ok  a different checkpoint is rejected as incomparable")

    # 15. MTP on: refused, then judged on accept length and drift only
    mtp_base = _run(mtp=True, prompts=[_prompt("p", 76, accept=3.4)])
    mtp_cand = _run("qpn2", "both", mtp=True, prompts=[_prompt("p", 76, accept=3.2)])
    assert mtp_base["launch"]["mtp"] is True
    report = compare_runs(mtp_base, mtp_cand, mtp_secondary=True)
    assert report["status"] == FAIL and report["cause"] == "draft", report
    checks += 1
    print("ok  MTP-on runs are judged on accept length only (--mtp-secondary)")

    # 16. the CLI refuses MTP-on runs without --mtp-secondary
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.json"
        cand_path = Path(tmp) / "cand.json"
        base_path.write_text(json.dumps(mtp_base))
        cand_path.write_text(json.dumps(mtp_cand))
        try:
            run({"baseline": str(base_path), "candidates": [("both", str(cand_path))]}, False)
        except UsageError as exc:
            assert "MTP" in str(exc), exc
        else:
            raise AssertionError("MTP-on run was not refused")
        code, payload = run(
            {"baseline": str(base_path), "candidates": [("both", str(cand_path))]}, True
        )
        assert code == 1 and payload["verdict"] == FAIL
    checks += 1
    print("ok  the CLI refuses MTP-on runs and judges them with --mtp-secondary")

    print(f"selftest: {checks} checks passed")
    return 0


# --- CLI -------------------------------------------------------------------

def _parse_candidates(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
            if not label:
                raise UsageError(f"empty label in {item!r}")
        else:
            label, path = Path(item).stem, item
        out.append((label, path))
    if not out:
        raise UsageError("at least one candidate is required")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", nargs="?", help="baseline run artifact (Marlin everywhere)")
    parser.add_argument("candidates", nargs="*", help="LABEL=PATH, or PATH")
    parser.add_argument("--json", type=Path, help="write the verdict as JSON here")
    parser.add_argument("--mtp-secondary", action="store_true",
                        help=(
                            "judge accept length and drift only; "
                            "token equality is not attributable"
                        ))
    parser.add_argument("--selftest", action="store_true", help="run the synthetic checks and exit")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    if not args.baseline:
        parser.error("a baseline artifact is required (or --selftest)")

    try:
        paths = {
            "baseline": args.baseline,
            "candidates": _parse_candidates(args.candidates),
        }
        code, payload = run(paths, args.mtp_secondary)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_report(paths, payload)
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
