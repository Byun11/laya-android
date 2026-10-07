"""AndroidControl shards -> step-level JSONL per official split.

python scripts/build_androidcontrol_dataset.py [--shards 0,1] [--workers 10]
Out: $LAYA_ANDROID_DATA/processed/androidcontrol/{train,val,test_idd,test_app_unseen,test_task_unseen,test_category_unseen}.jsonl
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import candidates as C  # noqa: E402
from data import android_control as AC  # noqa: E402

OUT = os.path.join(AC.DATA_ROOT, "processed", "androidcontrol")
HISTORY = 3
SUBSPLIT_FILES = {"IDD": "test_idd", "app_unseen": "test_app_unseen", "task_unseen": "test_task_unseen",
                  "category_unseen": "test_category_unseen"}


def history_str(op, action, cands, gold):
    if op in ("CLICK", "LONG_PRESS"):
        return "%s %s" % (op, cands[gold]["label"] or cands[gold]["class"] if gold is not None else "(unlabeled)")
    if op == "INPUT_TEXT":
        return 'INPUT_TEXT "%s"' % action.get("text", "")
    if op == "OPEN_APP":
        return "OPEN_APP %s" % action.get("app_name", "")
    return op


def episode_items(ep, split):
    T = len(ep["actions"])
    assert len(ep["trees"]) == T + 1, (ep["episode_id"], T, len(ep["trees"]))
    hist, items = [], []
    for k in range(T + 1):
        nodes = AC.parse_forest(ep["trees"][k])
        cands = C.build(nodes, ep["screen"])
        if k < T:
            a = ep["actions"][k]
            op = AC.canonical_op(a)
        else:
            a, op = {}, "DONE"  # synthetic terminal action on the final observation
        gold = C.ground(cands, a["x"], a["y"]) if op in ("CLICK", "LONG_PRESS") else None
        items.append({
            "episode_id": ep["episode_id"], "split": split, "step": k, "goal": ep["goal"],
            "step_instruction": ep["step_instructions"][k] if k < len(ep["step_instructions"]) else "",
            "history": hist[-HISTORY:], "candidates": cands,
            "operation_gold": op, "target_gold": gold,
            "target_groundable": gold is not None if op in ("CLICK", "LONG_PRESS") else None,
            "candidate_count": len(cands), "n_nodes": len(nodes),
            "click": [a["x"], a["y"]] if "x" in a else None,
            "payload": a.get("text", a.get("app_name")),
        })
        hist.append(history_str(op, a, cands, gold))
    return items


def work(shard):
    split_of, _ = AC.load_splits()
    path = os.path.join(OUT, "shards", "%02d.jsonl" % shard)
    n = empty = 0
    with open(path + ".tmp", "w", encoding="utf8") as f:
        for ep in AC.episodes(AC.shard_paths()[shard]):
            split = split_of.get(ep["episode_id"])
            if split is None:
                continue
            if not ep["actions"]:  # no demonstration to learn from
                empty += 1
                continue
            for it in episode_items(ep, split):
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
    os.replace(path + ".tmp", path)
    return shard, n, empty


def merge(shards):
    _, sub = AC.load_splits()
    outs = {name: open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf8")
            for name in ["train", "val", *SUBSPLIT_FILES.values()]}
    counts = {k: 0 for k in outs}
    for s in shards:
        with open(os.path.join(OUT, "shards", "%02d.jsonl" % s), encoding="utf8") as f:
            for line in f:
                it = json.loads(line)
                if it["split"] == "train":
                    names = ["train"]
                elif it["split"] == "validation":
                    names = ["val"]
                else:  # test subsplits overlap by design: an episode can land in several files
                    names = [v for k, v in SUBSPLIT_FILES.items() if it["episode_id"] in sub[k]]
                for nm in names:
                    outs[nm].write(line)
                    counts[nm] += 1
    for f in outs.values():
        f.close()
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=",".join(str(i) for i in range(20)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    shards = [int(s) for s in args.shards.split(",")]
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    if not args.merge_only:
        todo = [s for s in shards if not os.path.exists(os.path.join(OUT, "shards", "%02d.jsonl" % s))]
        with Pool(min(args.workers, max(1, len(todo)))) as pool:
            for s, n, empty in pool.imap_unordered(work, todo):
                print("shard %02d: %d steps, %d empty episodes skipped" % (s, n, empty), flush=True)
    print("merged", merge(shards))
