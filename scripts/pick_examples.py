"""Pick README examples from VALIDATION predictions by a fixed rule (no test data, no hand-picking).

Rule: val steps whose gold op is CLICK, groundable, with 8-20 candidates; shuffle with seed 0;
take the first step the model gets fully right (op + target) and the first it gets wrong.
Out: results/val/examples.json
"""
import gzip
import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402

steps = {}
with open(os.path.join(AC.DATA_ROOT, "processed", "androidcontrol", "val.jsonl"), encoding="utf8") as f:
    for line in f:
        it = json.loads(line)
        steps[(it["episode_id"], it["step"])] = it
with gzip.open(os.path.join(ROOT, "results", "val", "preds", "val.preds.jsonl.gz"), "rt", encoding="utf8") as f:
    preds = [json.loads(line) for line in f]
pool = [p for p in preds if p["operation_gold"] == "CLICK" and p["target_probs"] and 8 <= p["candidate_count"] <= 20]
random.Random(0).shuffle(pool)


def view(p):
    it = steps[(p["episode_id"], p["step"])]
    labels = [S.option_label(c) for c in it["candidates"]]
    ops = sorted(p["operation_probs"].items(), key=lambda kv: -kv[1])[:3]
    tp = p["target_probs"]
    top = sorted(range(len(tp)), key=lambda i: -tp[i])[:3]
    return {"episode_id": p["episode_id"], "step": p["step"], "goal": it["goal"], "history": it["history"],
            "candidates": labels, "gold": {"operation": "CLICK", "target": it["target_gold"],
                                           "target_label": labels[it["target_gold"]]},
            "pred_operation_top3": [[k, round(v, 3)] for k, v in ops],
            "pred_target_top3": [[i, labels[i], round(tp[i], 3)] for i in top]}


def correct(p):
    return max(p["operation_probs"], key=p["operation_probs"].get) == "CLICK" and \
        max(range(len(p["target_probs"])), key=p["target_probs"].__getitem__) == steps[(p["episode_id"], p["step"])]["target_gold"]


out = {"rule": " ".join(__doc__.strip().splitlines()[2:4]), "success": view(next(p for p in pool if correct(p))),
       "failure": view(next(p for p in pool if not correct(p)))}
with open(os.path.join(ROOT, "results", "val", "examples.json"), "w", encoding="utf8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
