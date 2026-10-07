"""Standardized latency benchmark. python scripts/benchmark_latency.py --out results/latency/rtx4090.json

Protocol: one GPU, fp32 weights + bf16 autocast (same as evaluation), batch size 1, 100 warm-up steps,
1000 measured steps sampled (seed 0) from test_idd inputs (labels unused). Every step asks both questions
(operation + target over all candidates) in one forward, as an agent would.
  forward_ms  collate + forward + cuda sync
  e2e_ms      tokenization/encoding of the step + forward + softmax
Also: forward_ms by candidate-count bucket, batched throughput (batch 32), peak VRAM.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import serialization as S  # noqa: E402
from eval_laya_android import BUCKETS, HEAD_MAX_LEN, MAX_LEN, PROC, load, logits_for  # noqa: E402

WARMUP, MEASURE, BATCH = 100, 1000, 32


def encode_both(tok, it):
    """Operation + target questions for any step with candidates (no gold used)."""
    from laya.common import render_options
    from laya.train import encode_item, encode_state, make_item, to_internal
    r = S.row(it)
    qs = dict(r["questions"])
    if it["candidates"] and "target" not in qs:
        qs["target"] = {"type": "choice", "instructions": S.Q_TARGET,
                        "criteria": {str(i): S.option_label(c) for i, c in enumerate(it["candidates"])}}
    state_ids = encode_state(tok, r["state"], MAX_LEN)
    encs = []
    for qid, q in qs.items():
        qi = to_internal(qid, q)
        dummy = [1.0] + [0.0] * (len(render_options(qi)) - 1)  # target vector is unused here
        item, reason = make_item(tok, qi, dummy, state_ids, HEAD_MAX_LEN, MAX_LEN)
        assert item is not None, reason
        encs.append(encode_item(tok, item, MAX_LEN, HEAD_MAX_LEN))
    return encs


def stats(xs):
    xs = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(xs)), 2), "p90": round(float(np.percentile(xs, 90)), 2),
            "p95": round(float(np.percentile(xs, 95)), 2), "mean": round(float(xs.mean()), 2)}


def bench(model_id, steps):
    model, tok, _ = load(model_id)
    for it in steps[:WARMUP]:
        logits_for(model, tok, encode_both(tok, it))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd, e2e, by_b = [], [], {b: [] for b, _, _ in BUCKETS}
    for it in steps[WARMUP:WARMUP + MEASURE]:
        t0 = time.perf_counter()
        encs = encode_both(tok, it)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lg = logits_for(model, tok, encs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for row, e in zip(lg, encs):  # softmax per question, as decoding would
            z = row[:len(e["markers"])]
            np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        t3 = time.perf_counter()
        fwd.append((t2 - t1) * 1000)
        e2e.append((t3 - t0) * 1000)
        b = next((n for n, lo, hi in BUCKETS if lo <= it["candidate_count"] <= hi), None)
        if b:
            by_b[b].append((t2 - t1) * 1000)
    vram_bs1 = torch.cuda.max_memory_allocated() / 2 ** 30

    # batched throughput: steps/s with BATCH steps (both questions each) per forward, encoding excluded
    encoded = [encode_both(tok, it) for it in steps[WARMUP:WARMUP + MEASURE]]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(encoded), BATCH):
        logits_for(model, tok, [e for encs in encoded[s:s + BATCH] for e in encs])
    torch.cuda.synchronize()
    thr = len(encoded) / (time.perf_counter() - t0)
    out = {"forward_ms": stats(fwd), "e2e_ms": stats(e2e),
           "forward_ms_by_candidates": {b: stats(v) if v else None for b, v in by_b.items()},
           "throughput_steps_per_s_batch%d" % BATCH: round(thr, 1),
           "peak_vram_gb_bs1": round(vram_bs1, 2),
           "peak_vram_gb_batch%d" % BATCH: round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    del model
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="multilingual,D:/laya-android/checkpoints/laya-android-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(PROC, "test_idd.jsonl"), encoding="utf8") as f:
        pool = [json.loads(line) for line in f]
    steps = random.Random(0).sample(pool, WARMUP + MEASURE)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "fp32 weights + bf16 autocast",
              "batch_size": 1, "warmup": WARMUP, "measured": MEASURE, "inputs": "test_idd, seed 0 (labels unused)",
              "questions_per_step": "operation + target (all candidates), one forward", "max_len": MAX_LEN,
              "head_max_len": HEAD_MAX_LEN, "models": {}}
    for m in args.models.split(","):
        name = "base_laya" if m == "multilingual" else "laya_android"
        report["models"][name] = bench(m, steps)
        print(name, json.dumps(report["models"][name]["forward_ms"]), json.dumps(report["models"][name]["e2e_ms"]), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
