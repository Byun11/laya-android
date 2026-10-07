"""Stage (and optionally upload) a Laya-Android checkpoint as a Hugging Face model repo.

python scripts/publish_hf.py --checkpoint D:/laya-android/checkpoints/laya-android-v0 \
    --repo <namespace>/laya-android --eval results/test/final_metrics.json [--eval ...] [--upload]

Without --upload it only writes $LAYA_ANDROID_DATA/hf_staging/<repo-name>/ for inspection. The staged folder loads with
`laya.load(path)`: model.safetensors, encoder/, tokenizer/, rl_agent_config.json, plus README.md
(from MODEL_CARD.md), figures/, LICENSE, NOTICE, laya_android_metadata.json and the given eval JSONs.
"""
import argparse
import json
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = {"repo": "convaiinnovations/laya", "subfolder": "multilingual",
        "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148", "laya_package": "0.3.28"}
REQUIRED = ["model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"]


def stage(checkpoint, repo, evals):
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(checkpoint, f))]
    if missing:
        raise SystemExit("checkpoint %s is missing %s" % (checkpoint, missing))
    out = os.path.join(os.environ.get("LAYA_ANDROID_DATA", "D:/laya-android"), "hf_staging", repo.split("/")[-1])
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in REQUIRED:
        src = os.path.join(checkpoint, f)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(out, f))
    shutil.copy2(os.path.join(ROOT, "MODEL_CARD.md"), os.path.join(out, "README.md"))
    for f in ("LICENSE", "NOTICE"):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(out, f))
    shutil.copytree(os.path.join(ROOT, "figures"), os.path.join(out, "figures"))  # images the card links to
    with open(os.path.join(checkpoint, "rl_agent_config.json"), encoding="utf8") as f:
        cfg = json.load(f)
    meta = {"base_model": BASE, "max_len": cfg.get("max_len"), "head_max_len": cfg.get("head_max_len"),
            "training": cfg.get("training"), "evaluations": [os.path.basename(e) for e in evals]}
    with open(os.path.join(out, "laya_android_metadata.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=1)
    os.makedirs(os.path.join(out, "eval"), exist_ok=True)
    for e in evals:
        with open(e, encoding="utf8") as f:
            report = json.load(f)
        report["model"] = repo  # drop the local checkpoint path
        with open(os.path.join(out, "eval", os.path.basename(e)), "w", encoding="utf8") as f:
            json.dump(report, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="<namespace>/laya-android")
    ap.add_argument("--eval", action="append", default=[])
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    out = stage(args.checkpoint, args.repo, args.eval)
    print("staged", out)
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.repo, repo_type="model",
                          commit_message="Laya-Android weights and model card")
        print("uploaded https://huggingface.co/%s" % args.repo)
