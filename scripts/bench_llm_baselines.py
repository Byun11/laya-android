"""Same-input generative baselines vs Laya-Android: latency and AndroidControl-High Type / Grounding.

python scripts/bench_llm_baselines.py --out results/baselines/same_input_1000.json
Subset: 1,000 steps sampled (seed 0) from the 8,444 official test steps; identical for every model.
Input for every model: goal + last 3 gold actions + the same accessibility-derived candidates (src/serialization.py).
LLMs (zero-shot, greedy, batch 1) must answer {"operation": ..., "target": index}. Laya-Android answers both questions
in one forward pass. Latency = tokenization + model call (+ generation), CUDA-synchronized, after 20 warm-up steps.
Scoring: unmodified InfiGUI-R1 judge via scripts/evaluate_androidcontrol.to_reference.
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "infigui_r1"))
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402
from evaluate_android_control import evaluate_android_control_action  # noqa: E402
from evaluate_androidcontrol import SPLITS, to_reference  # noqa: E402

N, WARMUP, SEED = 1000, 20, 0
OPS = list(S.OP_DESC)
LLMS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
        "google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "LiquidAI/LFM2.5-1.2B-Instruct"]  # newest small open models, 2026-10
SYSTEM = "You are an Android GUI agent. Given the user's goal, the previous actions and the UI elements on screen, choose the next action."


def load_subset(reference):
    ref = {}
    for line in json.load(open(reference, encoding="utf8")):
        for k, pam in enumerate(line["step_check_pams"]):
            ref[(line["episode"]["episode_id"], k)] = (pam, line["width"], line["height"])
    steps = {}
    for s in SPLITS:
        with open(os.path.join(AC.DATA_ROOT, "processed", "androidcontrol", s + ".jsonl"), encoding="utf8") as f:
            for line in f:
                it = json.loads(line)
                if it["operation_gold"] != "DONE":
                    steps[(it["episode_id"], it["step"])] = it
    keys = random.Random(SEED).sample(sorted(ref), N + WARMUP)
    return keys, ref, steps


def prompt(it):
    labels = [S.option_label(c) for c in it["candidates"]]
    ui = "\n".join("[%d] %s" % (i, l) for i, l in enumerate(labels)) or "(none)"
    ops = "\n".join("%s: %s" % kv for kv in S.OP_DESC.items())
    hist = "\n".join(it["history"]) or "(none)"
    return ("Goal: %s\n\nPrevious actions:\n%s\n\nUI elements:\n%s\n\nOperations:\n%s\n\n"
            'Answer with JSON only: {"operation": "<OPERATION>", "target": <element index or null>}' %
            (it["goal"], hist, ui, ops))


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    op = str(d.get("operation", "")).upper()
    tgt = d.get("target")
    try:
        tgt = int(tgt) if tgt is not None else None
    except (TypeError, ValueError):
        tgt = None
    return (op if op in OPS else None), tgt


def score(rows):
    """rows: [(op, target_box, pam, w, h)] -> Type, Grounding (InfiGUI rule)."""
    t = g = n_click = 0
    for op, box, pam, w, h in rows:
        pred = to_reference(op, box, w, h) if op else {"action": "none"}
        if pred["action"] == "none":
            type_ok, exact = False, False
        else:
            type_ok, exact = evaluate_android_control_action(pred, pam, w, h, w, h)
        t += bool(type_ok)
        g += bool(exact) and pred["action"] == "click"
        n_click += pam["action"] == "click"
    return round(100 * t / len(rows), 1), round(100 * g / n_click, 1)


def lat_stats(xs):
    xs = np.array(xs)
    return {"median": round(float(np.median(xs)), 1), "p95": round(float(np.percentile(xs, 95)), 1),
            "mean": round(float(xs.mean()), 1)}


def run_llm(name, keys, ref, steps):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except (ValueError, KeyError):  # multimodal checkpoints; used with text input only
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    model = model.cuda().eval()
    rows, lat, gen_tokens, in_tokens, fails = [], [], [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for i, key in enumerate(keys):
        it = steps[key]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(it)}]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      enable_thinking=False)  # no reasoning trace: the fastest setting for the LLM
        ids = enc["input_ids"].cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if i < WARMUP:
            continue
        op, tgt = parse(text)
        fails += op is None
        box = it["candidates"][tgt]["bounds"] if op in ("CLICK", "LONG_PRESS") and tgt is not None and 0 <= tgt < len(it["candidates"]) else None
        pam, w, h = ref[key]
        rows.append((op, box, pam, w, h))
        lat.append(dt)
        gen_tokens.append(out.shape[1] - ids.shape[1])
        in_tokens.append(ids.shape[1])
    typ, gr = score(rows)
    res = {"params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2), "type": typ, "grounding": gr,
           "unparseable": fails, "latency_ms": lat_stats(lat), "mean_generated_tokens": round(float(np.mean(gen_tokens)), 1),
           "mean_input_tokens": round(float(np.mean(in_tokens)), 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    del model
    torch.cuda.empty_cache()
    return res


def run_laya(keys, ref, steps, checkpoint):
    from benchmark_latency import encode_both
    from eval_laya_android import load, logits_for
    model, tok, cfg = load(checkpoint)
    pred = {}
    with gzip.open(os.path.join(ROOT, "results", "test", "final_predictions.jsonl.gz"), "rt", encoding="utf8") as f:
        for line in f:
            p = json.loads(line)
            pred[(p["episode_id"], p["step"])] = p
    lat = []
    torch.cuda.reset_peak_memory_stats()
    for i, key in enumerate(keys):  # latency with the same end-to-end definition (encode + forward)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits_for(model, tok, encode_both(tok, steps[key]))
        torch.cuda.synchronize()
        if i >= WARMUP:
            lat.append((time.perf_counter() - t0) * 1000)
    rows = []
    for key in keys[WARMUP:]:  # accuracy from the saved one-shot test predictions (no new test inference)
        p, it = pred[key], steps[key]
        a = p["pred_action"]
        box = None
        if a["action"] in ("click", "long_press") and a["coordinate"][0] >= 0:
            x, y = a["coordinate"]
            box = [x, y, x, y]
        pam, w, h = ref[key]
        rows.append((p["pred_operation"], box, pam, w, h))
    typ, gr = score(rows)
    res = {"params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2), "type": typ, "grounding": gr,
           "unparseable": 0, "latency_ms": lat_stats(lat), "mean_generated_tokens": 0,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    del model
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(AC.DATA_ROOT, "raw", "infigui", "android_control_test.json"))
    ap.add_argument("--checkpoint", default=os.path.join(AC.DATA_ROOT, "checkpoints", "laya-android-v0"))
    ap.add_argument("--models", default=",".join(LLMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--laya", action="store_true", help="measure Laya-Android (main venv with laya)")
    args = ap.parse_args()
    keys, ref, steps = load_subset(args.reference)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = json.load(open(args.out)) if os.path.exists(args.out) else {
        "subset": {"n": N, "warmup": WARMUP, "seed": SEED, "keys": keys[WARMUP:]}, "gpu": torch.cuda.get_device_name(0),
        "protocol": __doc__.strip(), "models": {}}
    if args.laya:
        report["models"]["Laya-Android (322M)"] = run_laya(keys, ref, steps, args.checkpoint)
        print("Laya-Android", json.dumps(report["models"]["Laya-Android (322M)"]), flush=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        sys.exit(0)
    import transformers
    report["llm_software"] = {"transformers": transformers.__version__, "torch": torch.__version__}
    for name in args.models.split(","):
        try:
            report["models"][name] = run_llm(name, keys, ref, steps)
        except Exception as ex:  # record and continue; a failed model is reported, not dropped silently
            report["models"][name] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
            torch.cuda.empty_cache()
        print(name, json.dumps(report["models"][name]), flush=True)
        with open(args.out, "w") as f:  # partial results survive a crash
            json.dump(report, f, indent=1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
