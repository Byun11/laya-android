"""Same-input generative baselines served with vLLM (Docker, Linux kernels) on the same RTX 4090.

python scripts/bench_vllm.py --out results/baselines/same_input_1000.json [--models a,b]
Same 1,000-step subset, prompt, parsing and InfiGUI-R1 scoring as scripts/bench_llm_baselines.py.
Each request: chat completion, temperature 0, max_tokens 32, thinking disabled, batch 1 (sequential).
Latency = client wall time of the HTTP request (server-side tokenization + prefill + decode + ~1-2 ms loopback).
Results are merged into the same JSON as Laya-Android (measured in-process with tokenization).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_llm_baselines import LLMS, SYSTEM, WARMUP, lat_stats, load_subset, parse, prompt, score  # noqa: E402
from data import android_control as AC  # noqa: E402

IMAGE = "vllm/vllm-openai:latest"
PORT = 8000
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def post(path, body, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start(model):
    subprocess.run(["docker", "rm", "-f", "laya-bench-vllm"], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", "laya-bench-vllm", "--gpus", "all", "--ipc=host", "-p", "%d:8000" % PORT,
           "-v", "%s:/root/.cache/huggingface" % HF_CACHE, "-e", "HF_TOKEN", IMAGE,
           "--model", model, "--max-model-len", "8192", "--gpu-memory-utilization", "0.85", "--max-num-seqs", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < 3600:  # first run downloads weights
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5)
            return
        except Exception:
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "laya-bench-vllm"],
                                   capture_output=True, text=True).stdout.strip()
            if state != "true":
                logs = subprocess.run(["docker", "logs", "--tail", "15", "laya-bench-vllm"], capture_output=True, text=True)
                raise RuntimeError("vLLM exited: " + (logs.stdout + logs.stderr)[-800:])
            time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


def stop():
    subprocess.run(["docker", "rm", "-f", "laya-bench-vllm"], capture_output=True)


def run(model, keys, ref, steps):
    start(model)
    try:
        info = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/version" % PORT, timeout=10).read())
        rows, lat, gen, inp, fails = [], [], [], [], 0
        for i, key in enumerate(keys):
            it = steps[key]
            body = {"model": model, "temperature": 0, "max_tokens": 32,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(it)}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.perf_counter()
            r = post("/v1/chat/completions", body)
            dt = (time.perf_counter() - t0) * 1000
            if i < WARMUP:
                continue
            op, tgt = parse(r["choices"][0]["message"]["content"] or "")
            fails += op is None
            box = (it["candidates"][tgt]["bounds"] if op in ("CLICK", "LONG_PRESS") and tgt is not None
                   and 0 <= tgt < len(it["candidates"]) else None)
            pam, w, h = ref[key]
            rows.append((op, box, pam, w, h))
            lat.append(dt)
            gen.append(r["usage"]["completion_tokens"])
            inp.append(r["usage"]["prompt_tokens"])
        typ, gr = score(rows)
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        return {"serving": "vLLM %s (Docker)" % info.get("version"), "type": typ, "grounding": gr, "unparseable": fails,
                "latency_ms": lat_stats(lat), "mean_generated_tokens": round(sum(gen) / len(gen), 1),
                "mean_input_tokens": round(sum(inp) / len(inp), 1),
                "gpu_memory_used_mb_incl_kv_cache_reservation": int(smi.splitlines()[0]) if smi else None}
    finally:
        stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(AC.DATA_ROOT, "raw", "infigui", "android_control_test.json"))
    ap.add_argument("--models", default=",".join(LLMS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keys, ref, steps = load_subset(args.reference)
    report = json.load(open(args.out))
    assert report["subset"]["keys"] == [list(k) for k in keys[WARMUP:]] or report["subset"]["keys"] == keys[WARMUP:]
    for name in args.models.split(","):
        try:
            report["models"][name] = run(name, keys, ref, steps)
        except Exception as ex:
            report["models"][name] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:600])}
        print(name, json.dumps(report["models"][name]), flush=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
