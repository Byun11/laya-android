"""Offline replay storyboards: Laya-Android's saved validation predictions drawn on the recorded AndroidControl screens.

python scripts/make_replay.py
The model never sees these images; it read the accessibility tree. Each frame is one step with gold history
(teacher forcing), exactly as evaluated. Episodes are picked by a fixed rule from VALIDATION only:
  val episodes with 4-8 real actions, ids sorted then shuffled with seed 0;
  "success" = first episode where every step (incl. DONE) is correct; "mixed" = first with >=1 wrong and >=50% right.
Out: figures/replay_{success,mixed}.png (every step of the episode), results/val/replay_episodes.json
"""
import gzip
import io
import json
import os
import random
import sys
import textwrap
from multiprocessing import Pool

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import serialization as S  # noqa: E402
from data import android_control as AC  # noqa: E402

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
BLUE, GREEN, INK, MUTED, BG = "#2a78d6", "#008300", "#0b0b0b", "#6f6e69", "#fcfcfb"
W, H, SCREEN_H = 1200, 900, 860


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def load_val():
    steps, preds = {}, {}
    with open(os.path.join(AC.DATA_ROOT, "processed", "androidcontrol", "val.jsonl"), encoding="utf8") as f:
        for line in f:
            it = json.loads(line)
            steps[(it["episode_id"], it["step"])] = it
    with gzip.open(os.path.join(ROOT, "results", "val", "preds", "val.preds.jsonl.gz"), "rt", encoding="utf8") as f:
        for line in f:
            p = json.loads(line)
            preds[(p["episode_id"], p["step"])] = p
    return steps, preds


def judge(it, p):
    """(pred_op, op_prob, top2, pred_target or None, target_prob, correct) under Policy Joint."""
    ops = sorted(p["operation_probs"].items(), key=lambda kv: -kv[1])
    op = ops[0][0]
    tgt = tp = None
    if p["target_probs"]:
        tgt = max(range(len(p["target_probs"])), key=p["target_probs"].__getitem__)
        tp = p["target_probs"][tgt]
    ok = op == it["operation_gold"]
    if it["operation_gold"] in ("CLICK", "LONG_PRESS") and it["target_groundable"]:
        ok = ok and tgt == it["target_gold"]
    return op, ops[0][1], ops[1], tgt, tp, ok


def pick(steps, preds):
    by_ep = {}
    for (e, k), it in steps.items():
        by_ep.setdefault(e, []).append(k)
    eps = sorted(e for e, ks in by_ep.items() if 4 <= len(ks) - 1 <= 8)  # real actions, excluding DONE
    random.Random(0).shuffle(eps)
    res = {}
    for e in eps:
        oks = [judge(steps[(e, k)], preds[(e, k)])[-1] for k in sorted(by_ep[e])]
        kind = "success" if all(oks) else "mixed" if sum(oks) >= len(oks) / 2 else None
        if kind and kind not in res:
            res[kind] = (e, oks)
    return res


def find_shard(i, wanted):
    hits = {}
    for rec in AC.records(AC.shard_paths()[i]):
        ex = AC._example(rec, skip=("screenshots", "accessibility_trees", "actions", "step_instructions", "goal"))
        if ex["episode_id"][0] in wanted:
            hits[ex["episode_id"][0]] = i
    return hits


def screenshots(shard, episode_id):
    for rec in AC.records(AC.shard_paths()[shard]):
        ex = AC._example(rec, skip=("accessibility_trees",))
        if ex["episode_id"][0] == episode_id:
            return [Image.open(io.BytesIO(b)).convert("RGB") for b in ex["screenshots"]]
    raise KeyError(episode_id)


def short(c, n=34):
    lab = S.option_label(c)
    return lab if len(lab) <= n else lab[:n - 1].rstrip() + "…"


def storyboard(shots, steps, preds, e, ks):
    """All steps of one episode as a grid of annotated screenshots (no step is left out)."""
    cols, pw, sh, cap, gap, pad = 4, 250, 520, 64, 18, 24
    rows = -(-len(ks) // cols)
    first = steps[(e, ks[0])]
    W_ = pad * 2 + cols * pw + (cols - 1) * gap
    H_ = 96 + rows * (sh + cap + gap) + 52
    canvas = Image.new("RGB", (W_, H_), BG)
    d = ImageDraw.Draw(canvas)
    goal = textwrap.wrap("Goal: " + first["goal"].strip(), 82)
    for j, line in enumerate(goal[:2]):
        d.text((pad, 14 + 26 * j), line, font=font(19, True), fill=INK)
    d.text((pad, 70), "Validation episode %d, every step shown. Blue = predicted element, green = gold." % e,
           font=font(14), fill=MUTED)
    for i, k in enumerate(ks):
        it, p = steps[(e, k)], preds[(e, k)]
        op, op_p, _op2, tgt, tp, ok = judge(it, p)
        img = shots[k]
        scale = sh / img.height
        shot = img.resize((round(img.width * scale), sh))
        dd = ImageDraw.Draw(shot)
        box = lambda c: [v * scale for v in c["bounds"]]  # noqa: E731
        if it["operation_gold"] in ("CLICK", "LONG_PRESS") and it["target_groundable"]:
            dd.rectangle(box(it["candidates"][it["target_gold"]]), outline=GREEN, width=5)
        if op in ("CLICK", "LONG_PRESS") and tgt is not None:
            dd.rectangle(box(it["candidates"][tgt]), outline=BLUE, width=3)
        x = pad + (i % cols) * (pw + gap)
        y = 96 + (i // cols) * (sh + cap + gap)
        canvas.paste(shot, (x + (pw - shot.width) // 2, y))
        d.rectangle([x + (pw - shot.width) // 2 - 1, y - 1, x + (pw + shot.width) // 2, y + sh], outline="#d9d8d2")
        mark, color = ("✓", GREEN) if ok else ("✗", "#e34948")
        d.text((x, y + sh + 8), "%d. %s %.2f" % (k + 1, op, op_p), font=font(16, True), fill=BLUE)
        d.text((x + pw - 22, y + sh + 6), mark, font=font(20, True), fill=color)
        if op in ("CLICK", "LONG_PRESS") and tgt is not None:
            sub = short(it["candidates"][tgt], 30)
        elif op == "INPUT_TEXT":
            sub = "text: from goal / separate component"
        elif not ok:
            sub = "gold: " + it["operation_gold"]
        else:
            sub = ""
        d.text((x, y + sh + 32), sub, font=font(13), fill=MUTED)
    d.text((pad, H_ - 40), "Offline replay on recorded AndroidControl screens with gold history. "
           "The model never sees these images; it reads the accessibility tree.", font=font(14), fill=MUTED)
    return canvas


def main():
    steps, preds = load_val()
    chosen = pick(steps, preds)
    wanted = {e for e, _ in chosen.values()}
    with Pool(10) as pool:
        where = {}
        for h in pool.starmap(find_shard, [(i, wanted) for i in range(20)]):
            where.update(h)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    meta = {"rule": __doc__.strip().splitlines()[4:6]}
    for kind, (e, oks) in chosen.items():
        shots = screenshots(where[e], e)
        ks = sorted(k for (ee, k) in steps if ee == e)
        out = os.path.join(ROOT, "figures", "replay_%s.png" % kind)
        storyboard(shots, steps, preds, e, ks).save(out, optimize=True)
        meta[kind] = {"episode_id": e, "goal": steps[(e, ks[0])]["goal"], "steps_correct": oks,
                      "size_kb": round(os.path.getsize(out) / 1024)}
        print(kind, meta[kind], flush=True)
    with open(os.path.join(ROOT, "results", "val", "replay_episodes.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
