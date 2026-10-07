"""Same-input generative baselines served by the local Ollama server (llama.cpp, GGUF) on the same RTX 4090.

python scripts/bench_ollama.py --out results/baselines/same_input_1000.json --models qwen3.5:9b-fit,gemma4:26b
Same 1,000-step subset, prompt, parsing and InfiGUI-R1 scoring as scripts/bench_llm_baselines.py.
Each request: /api/chat, think=false, temperature 0, num_predict 32, num_ctx 8192, sequential (batch 1).
Latency = client wall time per request with the model already loaded (warm-up excluded), ~1 ms loopback included.
Parameter count and quantization are read from /api/show.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_llm_baselines import SYSTEM, WARMUP, lat_stats, load_subset, parse, prompt, score  # noqa: E402
from data import android_control as AC  # noqa: E402

URL = "http://127.0.0.1:11434"


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(model, keys, ref, steps, think=False, num_predict=32):
    show = post("/api/show", {"model": model})
    det = show.get("details", {})
    rows, lat, gen, inp, fails = [], [], [], [], 0
    for i, key in enumerate(keys):
        it = steps[key]
        body = {"model": model, "stream": False, "think": think, "keep_alive": "15m",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(it)}],
                "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 8192}}
        t0 = time.perf_counter()
        try:
            r = post("/api/chat", body)
        except urllib.error.HTTPError as ex:  # e.g. a model that does not accept think=false
            if ex.code == 400 and i == 0:
                body.pop("think")
                r = post("/api/chat", body)
            else:
                raise
        dt = (time.perf_counter() - t0) * 1000
        if i < WARMUP:
            continue
        op, tgt = parse(r["message"].get("content") or "")
        fails += op is None
        box = (it["candidates"][tgt]["bounds"] if op in ("CLICK", "LONG_PRESS") and tgt is not None
               and 0 <= tgt < len(it["candidates"]) else None)
        pam, w, h = ref[key]
        rows.append((op, box, pam, w, h))
        lat.append(dt)
        gen.append(r.get("eval_count", 0))
        inp.append(r.get("prompt_eval_count", 0))
    post("/api/generate", {"model": model, "keep_alive": 0})  # unload before the next model
    typ, gr = score(rows)
    return {"serving": "Ollama %s (GGUF)" % json.loads(urllib.request.urlopen(URL + "/api/version").read())["version"],
            "params": det.get("parameter_size"), "quantization": det.get("quantization_level"), "family": det.get("family"),
            "type": typ, "grounding": gr, "unparseable": fails, "latency_ms": lat_stats(lat),
            "mean_generated_tokens": round(sum(gen) / len(gen), 1), "mean_input_tokens": round(sum(inp) / len(inp), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(AC.DATA_ROOT, "raw", "infigui", "android_control_test.json"))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", default="false", help='false, or a reasoning level such as "low" for reasoning models')
    ap.add_argument("--num-predict", type=int, default=32)
    args = ap.parse_args()
    think = False if args.think == "false" else args.think
    keys, ref, steps = load_subset(args.reference)
    report = json.load(open(args.out))
    assert [list(k) for k in report["subset"]["keys"]] == [list(k) for k in keys[WARMUP:]], "subset mismatch"
    for name in args.models.split(","):
        key = "ollama:" + name + ("" if think is False else " (think=%s)" % think)
        try:
            report["models"][key] = run(name, keys, ref, steps, think, args.num_predict)
            report["models"][key].update(think=think, num_predict=args.num_predict)
        except Exception as ex:
            report["models"][key] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:400])}
        print(key, json.dumps(report["models"][key]), flush=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
