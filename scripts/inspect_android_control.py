"""Dataset report from processed JSONL -> results/data/androidcontrol_stats.json.

Token lengths use the base model tokenizer on the exact Laya rows (serialization.row), on a seeded
sample of up to --token-sample steps per split (tokenizing every option of 90k steps is slow).
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402

PROC = os.path.join(AC.DATA_ROOT, "processed", "androidcontrol")
SPLITS = ["train", "val", "test_idd", "test_app_unseen", "test_task_unseen", "test_category_unseen"]
BUCKETS = [("<=5", 0, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, 10 ** 9)]


def bucket(n):
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def dist(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1), "p50": q(.5), "p90": q(.9),
            "p99": q(.99), "max": xs[-1]}


def split_stats(path, tok, token_sample):
    from laya.common import build_head, serialize_state
    from laya.train import to_internal

    eps = collections.Counter()
    ops = collections.Counter()
    nodes, cands, clicks, sample = [], [], [], []
    g = collections.Counter()
    with open(path, encoding="utf8") as f:
        for line in f:
            it = json.loads(line)
            eps[it["episode_id"]] += 1
            ops[it["operation_gold"]] += 1
            nodes.append(it["n_nodes"])
            cands.append(it["candidate_count"])
            if it["operation_gold"] in ("CLICK", "LONG_PRESS"):
                g["click_steps"] += 1
                if it["target_groundable"]:
                    g["groundable"] += 1
                    c = it["candidates"][it["target_gold"]]
                    g["gold_unlabeled"] += not c["label"]
                    g["gold_scroll_container_only"] += c["scrollable"] and not (c["clickable"] or c["long_clickable"])
                    clicks.append(it["candidate_count"])
            sample.append(it)
    rnd = random.Random(0)
    sample = rnd.sample(sample, min(token_sample, len(sample)))
    goal_t, state_t, head_t = [], [], []
    for it in sample:
        r = S.row(it)
        goal_t.append(len(tok(it["goal"], add_special_tokens=False)["input_ids"]))
        state_t.append(len(tok(serialize_state(r["state"]), add_special_tokens=False)["input_ids"]))
        if "target" in r["questions"]:
            ids, _m, _s = build_head(tok, to_internal("target", r["questions"]["target"]), 10 ** 6)
            head_t.append(len(ids))
    traj = list(eps.values())  # steps per episode incl. synthetic DONE
    nb = collections.Counter(bucket(n) for n in clicks)
    return {
        "episodes": len(eps), "steps": sum(eps.values()), "steps_excl_done": sum(eps.values()) - len(eps),
        "action_types": dict(ops.most_common()),
        "trajectory_length_excl_done": dist([t - 1 for t in traj]),
        "ui_nodes_per_screen": dist(nodes), "candidates_per_screen": dist(cands),
        "candidates_at_groundable_clicks": dist(clicks),
        "candidate_buckets_at_groundable_clicks": {name: round(nb[name] / max(1, len(clicks)), 4)
                                                   for name, _, _ in BUCKETS},
        "candidate_buckets_all_steps": {name: round(sum(bucket(n) == name for n in cands) / len(cands), 4)
                                        for name, _, _ in BUCKETS},
        "grounding": {**g, "coverage": round(g["groundable"] / max(1, g["click_steps"]), 4)},
        "tokens_sampled_steps": len(sample),
        "goal_tokens": dist(goal_t), "state_tokens": dist(state_t), "target_head_tokens_untruncated": dist(head_t),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--token-sample", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "data", "androidcontrol_stats.json"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from laya.train import resolve_checkpoint_dir
    tok = AutoTokenizer.from_pretrained(os.path.join(resolve_checkpoint_dir(args.base), "tokenizer"))
    report = {s: split_stats(os.path.join(PROC, s + ".jsonl"), tok, args.token_sample) for s in SPLITS}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["train"], indent=1))
