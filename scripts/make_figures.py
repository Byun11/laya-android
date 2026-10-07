"""Figures (PNG + SVG) and README tables from result files only; never runs a model.

python scripts/make_figures.py
Reads results/test/{final,base}_metrics.json, results/test/raw/*/test_*.preds.jsonl.gz, results/val/training_history.json,
reports/{base_laya_val,smoke_val}.json, results/latency/rtx4090.json, results/external_baselines.json.
Writes figures/*.png|svg and results/tables.md (every number shown in a figure also appears there).
"""
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
SPLITS = [("test_idd", "IDD"), ("test_app_unseen", "App-Unseen"), ("test_task_unseen", "Task-Unseen"),
          ("test_category_unseen", "Category-Unseen")]
BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, 10 ** 9)]
# dataviz reference palette; validated with validate_palette.js "#2a78d6,#eb6834" --mode light (all checks pass)
OURS, BASE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#6f6e69", "#e6e5df", "#fcfcfb"
SMALL_N = 100


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf8") as f:
        return json.load(f)


def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=160)
    plt.close(fig)


def grouped(ax, labels, base, ours):
    w = 0.38
    xs = range(len(labels))
    ax.bar([x - w / 2 for x in xs], base, w, color=BASE, edgecolor=SURFACE, linewidth=1.5, label="Base Laya (zero-shot)")
    bars = ax.bar([x + w / 2 for x in xs], ours, w, color=OURS, edgecolor=SURFACE, linewidth=1.5, label="Laya-Android")
    for b, v in zip(bars, ours):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.3f" % v, ha="center", va="bottom", fontsize=9, color=INK)
    for x, v in zip(xs, base):
        ax.text(x - w / 2, v + 0.015, "%.3f" % v, ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)


def pooled_buckets(model):
    """Target top-1 by candidate count over the union of test steps (subsplits overlap; dedup by episode/step)."""
    seen, hit = set(), {b: [0, 0] for b, _, _ in BUCKETS}
    for s, _ in SPLITS:
        with gzip.open(os.path.join(R, "test", "raw", model, s + ".preds.jsonl.gz"), "rt", encoding="utf8") as f:
            for line in f:
                p = json.loads(line)
                key = (p["episode_id"], p["step"])
                if key in seen or not p["target_probs"]:
                    continue
                seen.add(key)
                tp = p["target_probs"]
                b = next(n for n, lo, hi in BUCKETS if lo <= p["candidate_count"] <= hi)
                hit[b][0] += max(range(len(tp)), key=tp.__getitem__) == p["target_gold"]
                hit[b][1] += 1
    return {b: {"acc": c / n if n else None, "n": n} for b, (c, n) in hit.items()}


def main():
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "svg.fonttype": "none"})
    fin, base = load("results", "test", "final_metrics.json"), load("results", "test", "base_metrics.json")
    fi, bi = fin["internal"]["splits"], base["internal"]["splits"]
    hist = load("results", "val", "training_history.json")
    bval, sval = load("reports", "base_laya_val.json")["splits"]["val"], load("reports", "smoke_val.json")["splits"]["val"]
    lat = load("results", "latency", "rtx4090.json")
    ext = load("results", "external_baselines.json")

    # Figure 1: Policy Joint across the four official test subsplits
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    grouped(ax, [n for _, n in SPLITS], [bi[s]["joint_action_accuracy"] for s, _ in SPLITS],
            [fi[s]["joint_action_accuracy"] for s, _ in SPLITS])
    ax.set_ylim(0, 0.85)
    style(ax, "Policy Joint Accuracy", "AndroidControl test: base Laya vs. Laya-Android (Policy Joint)")
    save(fig, "androidcontrol_generalization")

    # Figure 2: target top-1 by candidate count
    pb, po = pooled_buckets("base_laya"), pooled_buckets("laya_android")
    labels = [b for b, _, _ in BUCKETS]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    grouped(ax, ["%s\nn=%d%s" % (b, po[b]["n"], "*" if po[b]["n"] < SMALL_N else "") for b in labels],
            [pb[b]["acc"] for b in labels], [po[b]["acc"] for b in labels])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("UI candidates on screen (test steps with a groundable click target)", color=INK)
    style(ax, "Target top-1", "Target selection vs. number of candidates (all test steps)")
    if any(po[b]["n"] < SMALL_N for b in labels):
        fig.text(0.01, 0.01, "* n < %d: high variance, read with care" % SMALL_N, color=MUTED, fontsize=8)
    save(fig, "candidate_scaling")

    # Figure 3: validation training progression (one series, selected epoch marked)
    best = hist["best"]["epoch"]
    pts = [("Base", bval["joint_action_accuracy"]), ("Smoke-15k", sval["joint_action_accuracy"])] + \
          [("Epoch %d" % e["epoch"], e["val"]["joint_action_accuracy"]) for e in hist["epochs"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(range(len(pts)), [v for _, v in pts], color=OURS, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    sel = 1 + best
    ax.plot([sel], [pts[sel][1]], marker="o", markersize=14, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.5)
    for i, (n, v) in enumerate(pts):
        ax.text(i, v + 0.03, "%.3f%s" % (v, "\nselected" if i == sel else ""), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([n for n, _ in pts])
    ax.set_ylim(0, 0.9)
    style(ax, "Validation Policy Joint", "Training progression (validation split, not test)")
    save(fig, "training_progression")

    # tables
    hi = fin["androidcontrol_high"]["all"]
    e = ext[0]
    T = ["<!-- generated by scripts/make_figures.py from results/ ; do not edit by hand -->", "",
         "### AndroidControl-High (InfiGUI-R1 evaluator, %d steps / %d episodes)" % (hi["steps"], hi["episodes"]), "",
         "| Model | Params | Input | Type | Grounding | Full SR |", "|---|---:|---|---:|---:|---:|",
         "| %s† | %s | Screenshot | %.1f | %.1f | %.1f |" % (e["model"], e["params"], e["metrics"]["type"],
                                                         e["metrics"]["grounding"], e["metrics"]["full_sr"]),
         "| Laya base (zero-shot) | 322M | Accessibility tree | %.1f | %.1f | N/A |" % (
             base["androidcontrol_high"]["all"]["type"], base["androidcontrol_high"]["all"]["grounding"]),
         "| **Laya-Android** | **322M** | **Accessibility tree** | **%.1f** | **%.1f** | **N/A** |" % (hi["type"], hi["grounding"]),
         "", "### Base Laya → Laya-Android (Policy Joint Accuracy, test)", "",
         "| Split | Base Laya | Laya-Android | Δ |", "|---|---:|---:|---:|"]
    for s, n in SPLITS:
        b, o = bi[s]["joint_action_accuracy"], fi[s]["joint_action_accuracy"]
        T.append("| %s | %.3f | **%.3f** | +%.1fpt |" % (n, b, o, 100 * (o - b)))
    T += ["", "### Per split (Laya-Android, test)", "",
          "| Split | Steps | Type | Grounding | Policy Joint | Target top-1 | Op macro-F1 | Candidate coverage | ECE (op) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, n in SPLITS:
        h, d = fin["androidcontrol_high"][s], fi[s]
        T.append("| %s | %d | %.1f | %.1f | %.3f | %.3f | %.3f | %.1f%% | %.3f |" % (
            n, h["steps"], h["type"], h["grounding"], d["joint_action_accuracy"], d["target_top1_accuracy"],
            d["operation_macro_f1"], 100 * d["grounding_coverage"], d["ece_operation"]))
    T += ["", "### Target top-1 by candidate count (union of test steps)", "",
          "| Candidates | n | Base Laya | Laya-Android |", "|---|---:|---:|---:|"]
    T += ["| %s | %d | %.3f | %.3f |" % (b, po[b]["n"], pb[b]["acc"], po[b]["acc"]) for b in labels]
    T += ["", "### Training progression (validation Policy Joint)", "", "| Checkpoint | Val Policy Joint |", "|---|---:|"]
    T += ["| %s%s | %.3f |" % (n, " (selected)" if i == sel else "", v) for i, (n, v) in enumerate(pts)]
    m = lat["models"]["laya_android"]
    T += ["", "### Latency (%s)" % lat["gpu"], "",
          "| Params | dtype | Batch | Median | p90 | p95 | Mean | End-to-end median | Peak VRAM (bs1) | Throughput (bs32) |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
          "| %.0fM | bf16 autocast | 1 | %.1f ms | %.1f ms | %.1f ms | %.1f ms | %.1f ms | %.2f GB | %.0f steps/s |" % (
              m["params_m"], m["forward_ms"]["median"], m["forward_ms"]["p90"], m["forward_ms"]["p95"],
              m["forward_ms"]["mean"], m["e2e_ms"]["median"], m["peak_vram_gb_bs1"], m["throughput_steps_per_s_batch32"])]
    ops = list(fi[SPLITS[0][0]]["op_per_class"])
    T += ["", "### Operation recall (test; n = gold support)", "", "| Operation | " + " | ".join(n for _, n in SPLITS) + " |",
          "|---|" + "---:|" * len(SPLITS)]
    for op in ops:
        T.append("| %s | " % op + " | ".join("%.2f (n=%d)" % (fi[s]["op_per_class"][op]["recall"], fi[s]["op_per_class"][op]["n_gold"])
                                            if op in fi[s]["op_per_class"] and fi[s]["op_per_class"][op]["n_gold"] else "–"
                                            for s, _ in SPLITS) + " |")
    with open(os.path.join(R, "tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
