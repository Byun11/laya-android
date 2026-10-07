"""TypeSafe Jev on the same 1,000-step subset, with the exact Laya-Android questions (Jev /v1/systemone wire format).

TYPESAFE_API_KEY=... python scripts/bench_jev.py --out results/baselines/jev_1000.json [--limit 5]
Request per step: {"state", "questions": {operation, target}, "model": "jev-1.13.0"}; the same state and questions
Laya-Android answers (target asked whenever the screen has candidates). Scored with the InfiGUI-R1 judge.
Also: operation ECE/Brier on the same steps for Jev and for Laya-Android (saved one-shot test probabilities).
Latency = client wall time of the HTTPS call (includes the internet round trip; not a same-hardware comparison).
Jev outputs are used for evaluation only, never for training.
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import serialization as S  # noqa: E402
from bench_llm_baselines import WARMUP, lat_stats, load_subset, score  # noqa: E402
from data import android_control as AC  # noqa: E402
from eval_laya_android import calib  # noqa: E402
from evaluate_androidcontrol import SPLITS  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
OPS = list(S.OP_DESC)


def request(it):
    r = S.row(it)
    qs = dict(r["questions"])
    if it["candidates"] and "target" not in qs:
        qs["target"] = {"type": "choice", "instructions": S.Q_TARGET,
                        "criteria": {str(i): S.option_label(c) for i, c in enumerate(it["candidates"])}}
    return {"state": r["state"], "questions": qs}


def call(body, key, model):
    req = urllib.request.Request(URL, data=json.dumps({**body, "model": model}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def laya_op_probs(keys):
    probs = {}
    for s in SPLITS:
        with gzip.open(os.path.join(ROOT, "results", "test", "raw", "laya_android", s + ".preds.jsonl.gz"), "rt", encoding="utf8") as f:
            for line in f:
                p = json.loads(line)
                probs[(p["episode_id"], p["step"])] = p["operation_probs"]
    return [(np.array([probs[k][o] for o in OPS]), k) for k in keys]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(AC.DATA_ROOT, "raw", "infigui", "android_control_test.json"))
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--limit", type=int, help="smoke test: only this many measured steps, nothing saved")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    assert key, "TYPESAFE_API_KEY not set"
    keys, ref, steps = load_subset(args.reference)
    keys = keys[:WARMUP + args.limit] if args.limit else keys
    rows, lat, cal, tok_in, tok_out, errors = [], [], [], 0, 0, 0
    per_step = []  # per-step probabilities, for reliability diagrams
    for i, k in enumerate(keys):
        it = steps[k]
        try:
            out, dt = call(request(it), key, args.model)
        except Exception as ex:  # keep going; failures are counted as wrong
            errors += 1
            if i >= WARMUP:
                pam, w, h = ref[k]
                rows.append((None, None, pam, w, h))
            if errors <= 3:
                print("error:", type(ex).__name__, str(ex)[:200], flush=True)
            continue
        if i < WARMUP:
            continue
        ans = out["answers"]
        op = ans["operation"]["choice"]
        tgt = ans.get("target", {}).get("choice")
        box = it["candidates"][int(tgt)]["bounds"] if op in ("CLICK", "LONG_PRESS") and tgt is not None else None
        pam, w, h = ref[k]
        rows.append((op if op in OPS else None, box, pam, w, h))
        lat.append(dt)
        p = ans["operation"].get("probabilities", {})
        cal.append((np.array([p.get(o, 0.0) for o in OPS]), OPS.index(it["operation_gold"])))
        per_step.append({"episode_id": k[0], "step": k[1], "operation_gold": it["operation_gold"],
                         "operation_probs": {o: p.get(o, 0.0) for o in OPS},
                         "target_gold": it["target_gold"] if it["target_groundable"] else None,
                         "target_probs": ans.get("target", {}).get("probabilities")})
        u = out.get("usage", {})
        tok_in += u.get("input_tokens", 0)
        tok_out += u.get("output_tokens", 0)
        if args.limit:
            print(i - WARMUP, op, tgt, round(dt), "ms", u, flush=True)
    typ, gr = score(rows)
    ece, brier = calib(cal)
    res = {"serving": "TypeSafe API (%s)" % args.model, "type": typ, "grounding": gr, "errors": errors,
           "latency_ms_api_roundtrip": lat_stats(lat) if lat else None, "ece_operation": ece, "brier_operation": brier,
           "usage_tokens": {"input": tok_in, "output": tok_out}, "measured_at": time.strftime("%Y-%m-%d %H:%M")}
    lk = keys[WARMUP:]
    lp = laya_op_probs(lk)
    l_ece, l_brier = calib([(p, OPS.index(steps[k]["operation_gold"])) for p, k in lp])
    res["laya_android_same_steps"] = {"ece_operation": l_ece, "brier_operation": l_brier}
    print(json.dumps(res), flush=True)
    if not args.limit:
        with gzip.open(args.out.replace(".json", "_steps.jsonl.gz"), "wt", encoding="utf8") as f:
            for r in per_step:
                f.write(json.dumps(r) + "\n")
    if not args.limit:  # own file: the Ollama run rewrites same_input_1000.json while it is running
        with open(args.out, "w") as f:
            json.dump({"subset_keys": [list(k) for k in lk], "models": {"jev:" + args.model: res}}, f, indent=1)
