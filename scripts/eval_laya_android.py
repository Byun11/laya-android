"""Offline AndroidControl evaluation, one harness for base and fine-tuned Laya.

python scripts/eval_laya_android.py --model D:/laya-android/checkpoints/smoke --out reports/x.json [--splits val] [--limit N] [--low-level]

Metrics per split:
  operation_accuracy  over all steps (incl. synthetic DONE)
  target_top1         over groundable CLICK/LONG_PRESS steps (gold op teacher-forced for the question)
  joint_accuracy      op correct AND (target correct for click steps); ungroundable click steps excluded
  grounding_coverage  groundable / all click steps
  ECE (15 bins, top-label) and Brier for both questions, with the checkpoint's temperatures
  operation macro-F1 (classes in gold or pred) + per-class precision/recall/F1;
  target_top1 by candidate-count bucket; items that did not fit max_len (counted wrong)
  latency: per-step wall time of one forward with both questions, batch 1, first --latency-n steps
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402

PROC = os.path.join(AC.DATA_ROOT, "processed", "androidcontrol")
BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, 10 ** 9)]
MAX_LEN, HEAD_MAX_LEN = 3072, 1536  # from reports/androidcontrol_stats.json: no candidate is dropped


def load(model):
    from laya.train import load_checkpoint
    m, tok, cfg = load_checkpoint(model)
    return m.cuda().eval(), tok, cfg


def encode(tok, it, max_len, head_max_len, low_level):
    """[(qid, encoded | None)] for one step, canonical option order."""
    from laya.train import encode_item, encode_state, make_item, target_from_expected, to_internal
    r = S.row(it, low_level)
    state_ids = encode_state(tok, r["state"], max_len)
    out = []
    for qid, q in r["questions"].items():
        qi = to_internal(qid, q)
        item, reason = make_item(tok, qi, target_from_expected(qi, r["expected"][qid]), state_ids, head_max_len, max_len)
        out.append((qid, encode_item(tok, item, max_len, head_max_len) if item else None))
    return out


@torch.no_grad()
def logits_for(model, tok, encs):
    from laya.common import collate_items
    batch = collate_items([encs], tok.pad_token_id)
    args = [batch[k].cuda() for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(*args)
    return logits.float().cpu().numpy()


def temperature(cfg, qtype, k):
    from laya.calibrate import temp_bucket
    return cfg.get("temperature_by_options", {}).get(temp_bucket(qtype, k), (cfg.get("temperature") or [1, 1, 1])[qtype])


def calib(probs_gold):
    """ECE (15 bins, top-label) and multiclass Brier from [(probs, gold_idx)]."""
    if not probs_gold:
        return None, None
    conf = np.array([p.max() for p, _ in probs_gold])
    hit = np.array([p.argmax() == g for p, g in probs_gold], dtype=float)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(conf[bins == b].mean() - hit[bins == b].mean()) * (bins == b).mean() for b in range(15) if (bins == b).any())
    brier = np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in probs_gold])
    return round(float(ece), 4), round(float(brier), 4)


def op_per_class(pairs, ops):
    """Per-class precision/recall/F1 and macro-F1 over classes present in gold or predictions."""
    per, f1s = {}, []
    for op in ops:
        tp = sum(g == op and p == op for g, p in pairs)
        n_gold = sum(g == op for g, _ in pairs)
        n_pred = sum(p == op for _, p in pairs)
        if not (n_gold or n_pred):
            continue
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per[op] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "n_gold": n_gold, "n_pred": n_pred}
    return {"operation_macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None, "op_per_class": per}


def evaluate_split(model, tok, cfg, path, limit=None, low_level=False, batch_size=32, latency_n=200, preds_path=None):
    steps = []
    with open(path, encoding="utf8") as f:
        for line in f:
            steps.append(json.loads(line))
            if limit and len(steps) >= limit:
                break
    ops = list(S.OP_DESC)
    flat = []  # (step_idx, qid, encoded)
    unfit = collections.Counter()
    for i, it in enumerate(steps):
        for qid, enc in encode(tok, it, MAX_LEN, HEAD_MAX_LEN, low_level):
            if enc is None:
                unfit[qid] += 1
            else:
                flat.append((i, qid, enc))
    order = sorted(range(len(flat)), key=lambda j: len(flat[j][2]["ids"]))  # length-sorted batches
    pred = {}
    for s in range(0, len(order), batch_size):
        chunk = [flat[j] for j in order[s:s + batch_size]]
        lg = logits_for(model, tok, [c[2] for c in chunk])
        for (i, qid, enc), row in zip(chunk, lg):
            k = len(enc["markers"])
            z = row[:k] / temperature(cfg, enc["qtype"], k)
            p = np.exp(z - z.max())
            pred[(i, qid)] = p / p.sum()

    m = collections.Counter()
    by_bucket = collections.defaultdict(lambda: [0, 0])
    op_pairs = []  # (gold, pred | None)
    cal = {"operation": [], "target": []}
    for i, it in enumerate(steps):
        gop = it["operation_gold"]
        p_op = pred.get((i, "operation"))
        pop = ops[int(p_op.argmax())] if p_op is not None else None
        op_ok = pop == gop
        op_pairs.append((gop, pop))
        if p_op is not None:
            cal["operation"].append((p_op, ops.index(gop)))
        m["steps"] += 1
        m["op_correct"] += op_ok
        is_click = gop in ("CLICK", "LONG_PRESS")
        if is_click:
            m["click_steps"] += 1
        if is_click and not it["target_groundable"]:
            continue  # target cannot be scored; excluded from joint, reported via coverage
        tgt_ok = True
        if is_click:
            m["groundable"] += 1
            p_t = pred.get((i, "target"))
            tgt_ok = p_t is not None and int(p_t.argmax()) == it["target_gold"]
            if p_t is not None:
                cal["target"].append((p_t, it["target_gold"]))
            m["target_correct"] += tgt_ok
            b = next(n for n, lo, hi in BUCKETS if lo <= it["candidate_count"] <= hi)
            by_bucket[b][0] += tgt_ok
            by_bucket[b][1] += 1
        m["joint_n"] += 1
        m["joint_correct"] += op_ok and tgt_ok

    if preds_path:  # raw per-step predictions, so later analysis never needs another pass over test
        with gzip.open(preds_path, "wt", encoding="utf8") as f:
            for i, it in enumerate(steps):
                p_op, p_t = pred.get((i, "operation")), pred.get((i, "target"))
                f.write(json.dumps({
                    "episode_id": it["episode_id"], "step": it["step"], "operation_gold": it["operation_gold"],
                    "target_gold": it["target_gold"], "target_groundable": it["target_groundable"],
                    "candidate_count": it["candidate_count"],
                    "operation_probs": None if p_op is None else dict(zip(ops, np.round(p_op.astype(float), 5).tolist())),
                    "target_probs": None if p_t is None else np.round(p_t.astype(float), 5).tolist()}) + "\n")

    # latency: both questions of a step in one forward, batch 1, after warmup
    lat = []
    for i in range(min(latency_n + 5, len(steps))):
        encs = [e for _, e in encode(tok, steps[i], MAX_LEN, HEAD_MAX_LEN, low_level) if e is not None]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits_for(model, tok, encs)
        torch.cuda.synchronize()
        if i >= 5:
            lat.append((time.perf_counter() - t0) * 1000)
    ece_op, brier_op = calib(cal["operation"])
    ece_t, brier_t = calib(cal["target"])
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "steps": m["steps"], "operation_accuracy": r(m["op_correct"], m["steps"]),
        "target_top1_accuracy": r(m["target_correct"], m["groundable"]),
        "joint_action_accuracy": r(m["joint_correct"], m["joint_n"]),
        "grounding_coverage": r(m["groundable"], m["click_steps"]),
        "ece_operation": ece_op, "brier_operation": brier_op, "ece_target": ece_t, "brier_target": brier_t,
        "latency_ms_median": round(float(np.median(lat)), 2) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2) if lat else None,
        "target_top1_by_candidates": {b: {"acc": r(*by_bucket[b]), "n": by_bucket[b][1]} for b, _, _ in BUCKETS},
        **op_per_class(op_pairs, ops),
        "unfit_items": dict(unfit),
    }


def main(default_model=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model, required=default_model is None)
    ap.add_argument("--splits", default="val,test_idd,test_app_unseen,test_task_unseen,test_category_unseen")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--low-level", action="store_true", help="oracle-low-level diagnostic (adds step instruction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds-dir", help="write <split>.preds.jsonl.gz with per-step probabilities")
    args = ap.parse_args()
    model, tok, cfg = load(args.model)
    report = {"model": args.model, "low_level": args.low_level, "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN,
              "limit": args.limit, "temperature": cfg.get("temperature"),
              "temperature_by_options": cfg.get("temperature_by_options"), "splits": {}}
    if args.preds_dir:
        os.makedirs(args.preds_dir, exist_ok=True)
    for s in args.splits.split(","):
        preds = os.path.join(args.preds_dir, s + ".preds.jsonl.gz") if args.preds_dir else None
        report["splits"][s] = evaluate_split(model, tok, cfg, os.path.join(PROC, s + ".jsonl"), args.limit,
                                             args.low_level, preds_path=preds)
        print(s, json.dumps({k: v for k, v in report["splits"][s].items() if not isinstance(v, dict)}), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
