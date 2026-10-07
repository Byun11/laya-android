"""AndroidControl-High Type / Grounding (InfiGUI-R1 evaluator) from saved one-shot test predictions. No inference.

python scripts/evaluate_androidcontrol.py --model laya_android --out results/test/final_metrics.json \
    --preds-out results/test/final_predictions.jsonl.gz
Definitions: docs/eval_protocol.md (frozen at commit f2c27d7 before these numbers were computed).
Inputs: results/preds/<model>/test_*.preds.jsonl.gz, results/test_<model>.json (internal metrics, same pass),
        $LAYA_ANDROID_DATA/processed/androidcontrol/test_*.jsonl (candidate boxes),
        reference test file android_control_test.json (InfiX-ai/android_control_test).
"""
import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "third_party", "infigui_r1"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate_android_control import evaluate_android_control_action  # noqa: E402  (unmodified reference judge)
from data import android_control as AC  # noqa: E402

PROC = os.path.join(AC.DATA_ROOT, "processed", "androidcontrol")
SPLITS = ["test_idd", "test_app_unseen", "test_task_unseen", "test_category_unseen"]
PROTOCOL_COMMIT = "f2c27d75a69d0fedc2f99cc39da158393e1034e5"
REF_ACTION = {"CLICK": "click", "LONG_PRESS": "long_press", "INPUT_TEXT": "type", "OPEN_APP": "open",
              "BACK": "system_button", "HOME": "system_button", "WAIT": "wait", "DONE": "terminate"}
SWIPE = {"SCROLL_UP": (0, -1), "SCROLL_DOWN": (0, 1), "SCROLL_LEFT": (-1, 0), "SCROLL_RIGHT": (1, 0)}


def to_reference(op, target_box, width, height):
    """Laya-Android operation (+ target box) -> reference action dict (docs/eval_protocol.md §3)."""
    if op in SWIPE:
        dx, dy = SWIPE[op]
        cx, cy = width / 2, height / 2
        return {"action": "swipe", "coordinate": [cx, cy], "coordinate2": [cx + dx * width / 4, cy + dy * height / 4]}
    a = {"action": REF_ACTION[op]}
    if op in ("CLICK", "LONG_PRESS"):
        a["coordinate"] = ([(target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2]
                           if target_box else [-10 * width, -10 * height])  # no saved target: far off-screen, guaranteed miss
    elif op in ("INPUT_TEXT", "OPEN_APP"):
        a["text"] = ""  # v0 generates no payloads; Full SR is therefore not reported
    elif op in ("BACK", "HOME"):
        a["button"] = "Back" if op == "BACK" else "Home"
    return a


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["laya_android", "base_laya"])
    ap.add_argument("--reference", default=os.path.join(AC.DATA_ROOT, "raw", "infigui", "android_control_test.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds-out", required=True)
    args = ap.parse_args()

    with open(args.reference, encoding="utf8") as f:
        ref = {}
        for line in json.load(f):
            for k, pam in enumerate(line["step_check_pams"]):
                ref[(line["episode"]["episode_id"], k)] = (pam, line["width"], line["height"])

    steps, members = {}, {s: set() for s in SPLITS}  # (ep, k) -> processed step; subsplit membership
    for s in SPLITS:
        with open(os.path.join(PROC, s + ".jsonl"), encoding="utf8") as f:
            for line in f:
                it = json.loads(line)
                if it["operation_gold"] == "DONE":
                    continue  # synthetic, not in the reference universe
                key = (it["episode_id"], it["step"])
                steps[key] = it
                members[s].add(key)
    preds = {}
    for s in SPLITS:
        with gzip.open(os.path.join(ROOT, "results", "preds", args.model, s + ".preds.jsonl.gz"), "rt", encoding="utf8") as f:
            for line in f:
                p = json.loads(line)
                if p["operation_gold"] != "DONE":
                    preds[(p["episode_id"], p["step"])] = p

    assert set(ref) == set(steps) == set(preds), (len(ref), len(steps), len(preds))
    gold_ref = lambda op: "swipe" if op in SWIPE else REF_ACTION[op]  # noqa: E731
    mism = sum(gold_ref(it["operation_gold"]) != ref[k][0]["action"] for k, it in steps.items())
    assert mism == 0, "gold action types disagree with the reference on %d steps" % mism

    rows = {}
    with gzip.open(args.preds_out, "wt", encoding="utf8") as out:
        for key in sorted(ref):
            pam, w, h = ref[key]
            p, it = preds[key], steps[key]
            op = max(p["operation_probs"], key=p["operation_probs"].get)
            box = None
            if op in ("CLICK", "LONG_PRESS") and p["target_probs"]:
                box = it["candidates"][int(np.argmax(p["target_probs"]))]["bounds"]
            pred = to_reference(op, box, w, h)
            type_ok, exact = evaluate_android_control_action(pred, pam, w, h, w, h)
            rows[key] = (type_ok, exact, pred["action"], pam["action"])
            out.write(json.dumps({"episode_id": key[0], "step": key[1], "pred_operation": op, "pred_action": pred,
                                  "gold_action": pam["action"], "type_match": bool(type_ok),
                                  "exact_match": bool(exact),
                                  "subsplits": [s for s in SPLITS if key in members[s]]}) + "\n")

    def score(keys):
        keys = list(keys)
        t = sum(rows[k][0] for k in keys)
        g_hit = sum(rows[k][1] and rows[k][2] == "click" for k in keys)
        g_n = sum(rows[k][3] == "click" for k in keys)
        return {"steps": len(keys), "episodes": len({k[0] for k in keys}),
                "type": round(100 * t / len(keys), 2), "grounding": round(100 * g_hit / g_n, 2) if g_n else None,
                "gold_clicks": g_n, "full_sr": None}

    internal = json.load(open(os.path.join(ROOT, "results", "test_%s.json" % args.model), encoding="utf8"))
    report = {
        "model": args.model, "protocol": "docs/eval_protocol.md", "protocol_commit": PROTOCOL_COMMIT,
        "code_commit": git_head(), "reference_evaluator": "InfiGUI-R1@a4fca17 evaluate_android_control_action",
        "full_sr_note": "N/A: v0 does not generate INPUT_TEXT text or OPEN_APP app names",
        "androidcontrol_low": "not evaluated (v0 trained with high-level goal only)",
        "androidcontrol_high": {"all": score(ref), **{s: score(members[s]) for s in SPLITS}},
        "internal": {"temperature": internal.get("temperature"), "splits": internal["splits"]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["androidcontrol_high"], indent=1))


if __name__ == "__main__":
    main()
