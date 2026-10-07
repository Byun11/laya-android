"""SFT (behavior cloning) of Laya on AndroidControl. python scripts/train_laya_android.py configs/smoke.yaml

Reuses laya.train.train_model (soft-CE over options) with two changes: bf16 autocast instead of the
hardcoded fp16, and per-epoch val evaluation keeping the best epoch by val joint accuracy.
Temperatures are fit on the val split afterwards (never on train or test).
Out: $LAYA_ANDROID_DATA/checkpoints/<name>/epoch{1..N}/ (uncalibrated, all kept), <name>/ = best epoch
     with val temperatures, and reports/train_<name>.json
"""
import json
import os
import random
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import laya.train as LT  # noqa: E402
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402
from eval_laya_android import HEAD_MAX_LEN, MAX_LEN, PROC, evaluate_split  # noqa: E402


def _forward_bf16(model, batch, device, amp, detach_encoder):
    args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        logits, _ = model(*args, detach_encoder=detach_encoder)
    return logits.float()


LT._forward = _forward_bf16  # ponytail: monkeypatch; laya 0.3.28 hardcodes fp16 at train.py:593

LOSS_CURVE = []  # mean loss per 500 micro-batches; laya only prints single micro-batch losses
_ce, _window = LT.soft_ce_loss, []


def _logged_ce(logits, target, mask):
    loss = _ce(logits, target, mask)
    if torch.is_grad_enabled():
        _window.append(loss.item())
        if len(_window) == 500:
            LOSS_CURVE.append(round(sum(_window) / 500, 4))
            print("loss window %d: %.4f" % (len(LOSS_CURVE), LOSS_CURVE[-1]), flush=True)
            _window.clear()
    return loss


LT.soft_ce_loss = _logged_ce


def load_steps(path, n_steps, seed):
    """All steps, or whole episodes in seeded random order until >= n_steps."""
    by_ep = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            it = json.loads(line)
            by_ep.setdefault(it["episode_id"], []).append(it)
    eps = sorted(by_ep)
    random.Random(seed).shuffle(eps)
    out = []
    for e in eps:
        if n_steps and len(out) >= n_steps:
            break
        out.extend(by_ep[e])
    return out, len(by_ep)


def main(cfg_path):
    with open(cfg_path) as f:
        c = yaml.safe_load(f)
    out_dir = os.path.join(AC.DATA_ROOT, "checkpoints", c["name"])
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "train_%s.json" % c["name"])
    steps, n_eps = load_steps(os.path.join(PROC, "train.jsonl"), c["train_steps"], c["seed"])
    model, tok, base_cfg = LT.load_checkpoint(c["base"])
    items, skipped = LT.items_from_rows(tok, (S.row(it) for it in steps), MAX_LEN, HEAD_MAX_LEN)
    print("steps %d (of %d train episodes), items %d, skipped %r" % (len(steps), n_eps, len(items), skipped), flush=True)

    tc = LT.TrainConfig(epochs=c["epochs"], micro_batch=c["micro_batch"], grad_accum=c["grad_accum"],
                        encoder_lr=c["encoder_lr"], head_lr=c["head_lr"], loss=c["loss"],
                        shuffle_options=tuple(c["shuffle_options"]), seed=c["seed"],
                        gradient_checkpointing=c["gradient_checkpointing"], max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    dev = torch.device("cuda")
    val_path = os.path.join(PROC, "val.jsonl")
    log = {"config": c, "train_steps": len(steps), "train_items": len(items), "skipped": skipped, "epochs": []}
    best = {"joint": -1.0}

    def on_epoch_end(epoch, loss):
        model.eval()
        v = evaluate_split(model, tok, base_cfg, val_path, latency_n=0)
        model.train()
        LT.save_checkpoint(model, tok, dict(base_cfg, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, fine_tuned=True,
                                            training={"laya_android": c, "epoch": epoch + 1, "uncalibrated": True}),
                           os.path.join(out_dir, "epoch%d" % (epoch + 1)))
        log["epochs"].append({"epoch": epoch + 1, "train_loss": loss, "val": v})
        with open(report_path, "w") as f:  # partial log survives a crash
            json.dump(log, f, indent=1)
        print("epoch %d loss %.4f val op %.4f macroF1 %.4f target %.4f joint %.4f" % (
            epoch + 1, loss, v["operation_accuracy"], v["operation_macro_f1"], v["target_top1_accuracy"],
            v["joint_action_accuracy"]), flush=True)
        if v["joint_action_accuracy"] > best["joint"]:  # selection: val joint only
            best.update(joint=v["joint_action_accuracy"], epoch=epoch + 1)

    t0 = time.time()
    LT.train_model(model, tok, items, tc, dev, MAX_LEN, HEAD_MAX_LEN, on_epoch_end=on_epoch_end)
    log["train_hours"] = round((time.time() - t0) / 3600, 3)
    log["loss_curve_per_500_microbatches"] = LOSS_CURVE
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(out_dir, "epoch%d" % best["epoch"], "model.safetensors")))
    log["best"] = best

    # temperatures on val only
    val_steps, _ = load_steps(val_path, None, 0)
    val_items, _ = LT.items_from_rows(tok, (S.row(it) for it in val_steps), MAX_LEN, HEAD_MAX_LEN)
    from laya.calibrate import fit_temperature_map
    fitted = fit_temperature_map(LT.calibration_records(model, tok, val_items, dev, MAX_LEN, HEAD_MAX_LEN))
    out_cfg = dict(base_cfg, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, fine_tuned=True,
                   temperature=fitted["temperature"], temperature_by_options=fitted["temperature_by_options"])
    out_cfg["training"] = {"laya_android": c, "best_epoch": best["epoch"], "train_hours": log["train_hours"],
                           "hardware": torch.cuda.get_device_name(0)}
    LT.save_checkpoint(model, tok, out_cfg, out_dir)
    log["temperature"] = fitted["temperature"]
    log["hardware"] = torch.cuda.get_device_name(0)
    with open(report_path, "w") as f:
        json.dump(log, f, indent=1)
    print("saved", out_dir, "best epoch", best["epoch"], "train hours", log["train_hours"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
