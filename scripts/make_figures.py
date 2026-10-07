"""Figures (PNG + SVG) and README tables from result files only; never runs a model.

python scripts/make_figures.py
Reads results/test/{final,base}_metrics.json, results/test/raw/*/test_*.preds.jsonl.gz, results/val/training_history.json,
results/val/{base_laya_val,smoke_val}.json, results/latency/rtx4090.json, results/external_baselines.json.
Writes figures/*.png|svg, results/tables.md, and rewrites every <!-- table: NAME --> block in the docs.
"""
import gzip
import json
import os
import re

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


def draw_overview(ex, ms, gpu):
    """Horizontal flow: inputs -> Laya-Android -> operation / target, with real numbers."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=MUTED, fill=SURFACE, title_color=INK, mono=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.4,
                                    edgecolor=edge, facecolor=fill))
        ax.text(x + 0.18, y + h - 0.2, title, ha="left", va="top", fontsize=10.5, color=title_color, weight="bold")
        ax.text(x + 0.18, y + h - 0.6, body, ha="left", va="top", fontsize=8.6, color=INK,
                family="monospace" if mono else None, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4))

    import textwrap
    goal = textwrap.wrap(ex["goal"], 40)
    goal = goal[:2] if len(goal) <= 2 else [goal[0], goal[1] + " …"]
    hist = " → ".join(h[:22] for h in ex["history"][-2:])
    box(0.1, 2.55, 4.3, 1.5, "Goal + recent actions",
        "\n".join(goal + [hist]), mono=False)
    cands = ex["candidates"]
    shown = ["[%d] %s" % (i, c[:24]) for i, c in enumerate(cands[:6])]
    box(0.1, 0.1, 4.3, 2.25, "Accessibility UI candidates (%d)" % len(cands), "\n".join(shown) + "\n…")
    box(5.15, 1.15, 2.4, 1.9, "Laya-Android", "322M encoder\none forward pass\nno text generation\nno screenshot",
        edge=OURS, fill="#eaf2fc", title_color=OURS, mono=False)
    arrow(4.45, 3.2, 5.1, 2.5)
    arrow(4.45, 1.2, 5.1, 1.7)
    (op, p_op), = ex["pred_operation_top3"][:1]
    t_idx, t_lab, p_t = ex["pred_target_top3"][0]
    box(8.3, 2.55, 3.6, 1.5, "Operation", "%-12s %.3f\n%-12s %.3f" % (op, p_op, *ex["pred_operation_top3"][1]))
    box(8.3, 0.65, 3.6, 1.5, "Target", "[%d] %-14s %.3f\n[%d] %-14s %.3f" % (
        t_idx, t_lab.split(" (")[0][:14], p_t, ex["pred_target_top3"][1][0],
        ex["pred_target_top3"][1][1].split(" (")[0][:14], ex["pred_target_top3"][1][2]))
    arrow(7.6, 2.5, 8.25, 3.2)
    arrow(7.6, 1.7, 8.25, 1.4)
    ax.text(6.35, 0.75, "~%.0f ms / decision\n(%s, batch 1)" % (ms, gpu.replace("NVIDIA GeForce ", "")),
            ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(8.3, 0.35, "real validation step; gold = %s [%d]" % (ex["gold"]["operation"], ex["gold"]["target"]),
            ha="left", va="top", fontsize=8, color=MUTED)
    save(fig, "overview")


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


SERVING_COLOR = {"Laya-Android (local)": OURS, "vLLM bf16": BASE, "Ollama GGUF": "#1baf7a", "TypeSafe API": "#eda100"}
DOC_FILES = ["README.md", "MODEL_CARD.md", "docs/results.md", "docs/benchmark.md", "docs/training.md"]


def tidy_size(p):
    """'752.39M' -> '0.8B', '25.2B' -> '25B', '4.2B' -> '4.2B'."""
    m = re.match(r"([0-9.]+)([MB])", p or "")
    if not m:
        return p or ""
    b = float(m.group(1)) / (1000 if m.group(2) == "M" else 1)
    return ("%.0fB" % b) if b >= 10 else ("%.1fB" % b)


def short_name(key):
    name = key.split(":", 1)[1] if key.startswith(("ollama:", "jev:")) else key
    return name.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def baseline_rows():
    """Valid same-input results: dicts with name, params, serving, type, grounding, latency (median/p95)."""
    if not os.path.exists(os.path.join(R, "baselines", "same_input_1000.json")):
        return [], []
    models = dict(load("results", "baselines", "same_input_1000.json")["models"])
    if os.path.exists(os.path.join(R, "baselines", "jev_1000.json")):
        models.update(load("results", "baselines", "jev_1000.json")["models"])
    rows, excluded = [], []
    for key, m in models.items():
        if "error" in m or m.get("unparseable", 0) > 500 or m.get("errors", 0) > 500:
            excluded.append((short_name(key), m.get("error", "most answers unparseable (reasoning cut by the token limit)")[:90]))
            continue
        if key.startswith("Laya-Android"):
            serving, params, lat = "Laya-Android (local)", "322M", m["latency_ms"]
        elif key.startswith("jev:"):
            serving, params, lat = "TypeSafe API", "undisclosed", m["latency_ms_api_roundtrip"]
        elif key.startswith("ollama:"):
            serving, params, lat = "Ollama GGUF", "%s (%s)" % (tidy_size(m.get("params")), m.get("quantization")), m["latency_ms"]
        else:
            size = re.search(r"-(E?[0-9.]+B)", key)
            serving, params, lat = "vLLM bf16", "%s (bf16)" % (size.group(1) if size else "?"), m["latency_ms"]
        name = "Laya-Android" if key.startswith("Laya") else short_name(key)
        if key.startswith("ollama:"):  # canonical name from the tag's model line + reported size (not the local tag)
            base = key.split(":")[1]
            fam = {"qwen3.5": "Qwen3.5", "qwen3.8": "Qwen3.8", "gemma4": "Gemma 4", "ministral-3": "Ministral 3",
                   "gpt-oss": "gpt-oss"}.get(base, base)
            tagsize = key.split(":")[2] if key.count(":") >= 2 else ""
            name = "%s %s" % (fam, tagsize.upper() if re.fullmatch(r"e\d+b", tagsize) else tidy_size(m.get("params")))
        if serving == "vLLM bf16":
            nm = key.split("/")[-1].replace("-it", "").replace("-Instruct", "")
            nm = re.sub(r"^gemma-4", "Gemma 4", nm)
            name = re.sub(r"-(E?[0-9.]+B)$", lambda mm: " " + mm.group(1), nm)
        rows.append({"name": name, "params": params,
                     "serving": serving, "type": m["type"], "grounding": m["grounding"], "lat": lat, "raw": m})
    rows.sort(key=lambda r: (r["serving"] != "Laya-Android (local)", -r["type"]))
    return rows, excluded


def draw_comparison(rows):
    """Leaderboard: one row per model (sorted by Type), accuracy left, latency right; only Laya-Android in color."""
    rows = sorted(rows, key=lambda r: (r["name"] != "Laya-Android", -r["type"]))
    tag = {"Laya-Android (local)": "322M · local", "vLLM bf16": "vLLM bf16", "Ollama GGUF": "Ollama 4-bit",
           "TypeSafe API": "API"}
    labels = ["%s  (%s)" % (r["name"], tag[r["serving"]]) for r in rows]
    colors = [OURS if r["name"] == "Laya-Android" else "#b9b8b1" for r in rows]
    n = len(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 0.42 * n + 1.6), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.06})
    y = list(range(n))[::-1]
    a1.barh(y, [r["type"] for r in rows], height=0.62, color=colors, edgecolor=SURFACE)
    for yy, r in zip(y, rows):
        a1.text(r["type"] + 0.8, yy, "%.1f" % r["type"], va="center", fontsize=9,
                color=INK, weight="bold" if r["name"] == "Laya-Android" else "normal")
    a1.set_xlim(0, 100)
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=9.5)
    for lbl in a1.get_yticklabels():
        if lbl.get_text().startswith("Laya-Android"):
            lbl.set_weight("bold")
            lbl.set_color(OURS)
    style(a1, "", "Accuracy: AndroidControl-High Type")
    a1.yaxis.grid(False)
    a1.xaxis.grid(True, color=GRID, linewidth=0.8)
    laya = next(r for r in rows if r["name"] == "Laya-Android")["lat"]["median"]
    med = [r["lat"]["median"] for r in rows]
    a2.barh(y, med, height=0.62, color=colors, edgecolor=SURFACE)
    a2.errorbar(med, y, xerr=[[0] * n, [r["lat"]["p95"] - r["lat"]["median"] for r in rows]], fmt="none",
                ecolor=MUTED, capsize=3, linewidth=1)
    for yy, r in zip(y, rows):
        txt = "%.0f ms" % r["lat"]["median"] if r["name"] == "Laya-Android" else \
            "%.0f ms  · %.1f× slower" % (r["lat"]["median"], r["lat"]["median"] / laya)
        a2.text(r["lat"]["p95"] + max(med) * 0.02, yy, txt, va="center", fontsize=9,
                color=OURS if r["name"] == "Laya-Android" else INK, weight="bold" if r["name"] == "Laya-Android" else "normal")
    a2.set_xlim(0, max(r["lat"]["p95"] for r in rows) * 1.45)
    style(a2, "", "Latency per decision (median, whisker = p95)")
    a2.yaxis.grid(False)
    a2.xaxis.grid(True, color=GRID, linewidth=0.8)
    a2.set_xlabel("ms (RTX 4090, batch 1; API = network round trip)", color=MUTED, fontsize=9)
    a1.set_xlabel("Type accuracy, 1,000 test steps", color=MUTED, fontsize=9)
    for a in (a1, a2):
        a.tick_params(axis="y", length=0)
    for ext in ("png", "svg"):  # bbox="tight" keeps the long model names; tight_layout cannot with shared axes
        fig.savefig(os.path.join(FIG, "efficiency_comparison.%s" % ext), dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def draw_scatter(rows):
    """Plain scatter: latency (log) vs Type; marker shape = serving; labels placed without overlap."""
    import math
    markers = {"Laya-Android (local)": "o", "vLLM bf16": "s", "Ollama GGUF": "^", "TypeSafe API": "D"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for serving, mk in markers.items():
        pts = [r for r in rows if r["serving"] == serving]
        if not pts:
            continue
        ours = serving.startswith("Laya")
        ax.scatter([r["lat"]["median"] for r in pts], [r["type"] for r in pts], marker=mk, s=70 if ours else 46,
                   color=OURS if ours else "#8f8e88", edgecolor=SURFACE, linewidth=1, zorder=3,
                   label={"Laya-Android (local)": "Laya-Android (in-process)", "TypeSafe API": "TypeSafe Jev (API round trip)"
                          }.get(serving, serving))
    from adjustText import adjust_text
    texts = [ax.text(r["lat"]["median"], r["type"], r["name"], fontsize=8,
                     color=OURS if r["name"] == "Laya-Android" else INK) for r in rows]
    lats = [r["lat"]["median"] for r in rows]
    ax.set_xscale("log")
    ticks = [t for t in (20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000) if min(lats) / 1.5 <= t <= max(lats) * 1.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(lats) / 1.4, max(lats) * 1.6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Median latency per decision (ms, log scale), RTX 4090, batch 1", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    style(ax, "Type accuracy (%)", "Type accuracy vs. latency (AndroidControl-High, 1,000 test steps)")
    adjust_text(texts, ax=ax, expand=(1.15, 1.4), arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
    save(fig, "efficiency_scatter")


def sync_docs(sections):
    """Rewrite every <!-- table: NAME --> ... <!-- /table --> block in the docs with the freshly generated table."""
    import re
    for f in DOC_FILES:
        path = os.path.join(ROOT, f)
        s = open(path, encoding="utf8").read()
        s2 = re.sub(r"(<!-- table: (.+?) -->\n).*?(\n<!-- /table -->)",
                    lambda m: m.group(1) + sections[m.group(2)] + m.group(3) if m.group(2) in sections else m.group(0),
                    s, flags=re.S)
        if s2 != s:
            open(path, "w", encoding="utf8").write(s2)


def main():
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "svg.fonttype": "none"})
    fin, base = load("results", "test", "final_metrics.json"), load("results", "test", "base_metrics.json")
    fi, bi = fin["internal"]["splits"], base["internal"]["splits"]
    hist = load("results", "val", "training_history.json")
    bval, sval = load("results", "val", "base_laya_val.json")["splits"]["val"], load("results", "val", "smoke_val.json")["splits"]["val"]
    lat = load("results", "latency", "rtx4090.json")
    ext = load("results", "external_baselines.json")

    # Figure 1: Policy Joint across the four official test subsplits
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    grouped(ax, [n for _, n in SPLITS], [bi[s]["joint_action_accuracy"] for s, _ in SPLITS],
            [fi[s]["joint_action_accuracy"] for s, _ in SPLITS])
    ax.set_ylim(0, 0.85)
    style(ax, "Policy Joint Accuracy", "AndroidControl test: base Laya vs. Laya-Android (Policy Joint)")
    save(fig, "androidcontrol_generalization")

    # Figure 2: candidate count -> accuracy (left) and latency (right); two panels, one y-scale each
    pb, po = pooled_buckets("base_laya"), pooled_buckets("laya_android")
    labels = [b for b, _, _ in BUCKETS]
    lc = lat["models"]["laya_android"]["forward_ms_by_candidates"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    grouped(ax, ["%s\nn=%d%s" % (b, po[b]["n"], "*" if po[b]["n"] < SMALL_N else "") for b in labels],
            [pb[b]["acc"] for b in labels], [po[b]["acc"] for b in labels])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("UI candidates on screen", color=INK)
    style(ax, "Target top-1 (test)", "Accuracy vs. candidates")
    xs = range(len(labels))
    med = [lc[b]["median"] for b in labels]
    ax2.bar(xs, med, 0.55, color=OURS, edgecolor=SURFACE, linewidth=1.5)
    ax2.errorbar(xs, med, yerr=[[0] * len(labels), [lc[b]["p95"] - lc[b]["median"] for b in labels]],
                 fmt="none", ecolor=MUTED, capsize=4, linewidth=1)
    for x, b in zip(xs, labels):
        ax2.text(x, lc[b]["p95"] + 1.5, "%.0f" % lc[b]["median"], ha="center", fontsize=9, color=INK)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(["%s\nn=%d%s" % (b, lc[b]["n"], "*" if lc[b]["n"] < SMALL_N else "") for b in labels])
    ax2.set_ylim(0, max(lc[b]["p95"] for b in labels) * 1.25)
    ax2.set_xlabel("UI candidates on screen", color=INK)
    style(ax2, "ms per decision (batch 1)", "Latency vs. candidates (median, whisker = p95)")
    fig.text(0.01, 0.01, "* n < %d: high variance. Left: union of test steps with a groundable click. "
             "Right: %d sampled test_idd steps." % (SMALL_N, lat["measured"]), color=MUTED, fontsize=8)
    save(fig, "candidate_scaling")

    # Figure 0: overview, drawn from a real validation example and the measured latency
    ex = load("results", "val", "examples.json")["success"]
    draw_overview(ex, lat["models"]["laya_android"]["forward_ms"]["median"], lat["gpu"])

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
         "", "### AndroidControl-High: base vs fine-tuned (official evaluator)", "",
         "| Model | Params | Type | Grounding | Full SR |", "|---|---:|---:|---:|---:|",
         "| Laya base (zero-shot) | 322M | %.1f | %.1f | N/A |" % (base["androidcontrol_high"]["all"]["type"],
                                                             base["androidcontrol_high"]["all"]["grounding"]),
         "| **Laya-Android** | **322M** | **%.1f** | **%.1f** | **N/A** |" % (hi["type"], hi["grounding"]),
         "| Δ | | +%.1f | +%.1f | |" % (hi["type"] - base["androidcontrol_high"]["all"]["type"],
                                       hi["grounding"] - base["androidcontrol_high"]["all"]["grounding"]),
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
    T += ["", "### Latency", "", "| Hardware | Precision | Median | p95 | Peak VRAM |", "|---|---|---:|---:|---:|",
          "| %s | bf16 autocast | %.1f ms | %.1f ms | %.2f GB |" % (lat["gpu"].replace("NVIDIA GeForce ", ""),
              m["forward_ms"]["median"], m["forward_ms"]["p95"], m["peak_vram_allocated_gb_bs1"])]
    T += ["", "### Latency by candidate count (batch 1)", "", "| Candidates | n | Median | p95 |", "|---|---:|---:|---:|"]
    T += ["| %s | %d | %.1f ms | %.1f ms |" % (b, v["n"], v["median"], v["p95"]) for b, v in m["forward_ms_by_candidates"].items()]
    T += ["", "### Latency by input length (batch 1, longest sequence of the step)", "",
          "| Input tokens | n | Median | p95 |", "|---|---:|---:|---:|"]
    T += ["| %s | %d | %.1f ms | %.1f ms |" % (b, v["n"], v["median"], v["p95"]) for b, v in m["forward_ms_by_input_tokens"].items() if v]
    T += ["", "### Batch size (steps per forward, inputs unsorted)", "",
          "| Batch | Median batch latency | p95 | Throughput | Peak VRAM allocated | Peak VRAM reserved |",
          "|---:|---:|---:|---:|---:|---:|"]
    T += ["| %s | %.1f ms | %.1f ms | %.1f steps/s | %.2f GB | %.2f GB |" % (
        bs, v["batch_ms"]["median"], v["batch_ms"]["p95"], v["throughput_steps_per_s"], v["peak_vram_allocated_gb"],
        v["peak_vram_reserved_gb"]) for bs, v in m["batch_sweep"].items()]
    ops = list(fi[SPLITS[0][0]]["op_per_class"])
    T += ["", "### Operation recall (test; n = gold support)", "", "| Operation | " + " | ".join(n for _, n in SPLITS) + " |",
          "|---|" + "---:|" * len(SPLITS)]
    for op in ops:
        T.append("| %s | " % op + " | ".join("%.2f (n=%d)" % (fi[s]["op_per_class"][op]["recall"], fi[s]["op_per_class"][op]["n_gold"])
                                            if op in fi[s]["op_per_class"] and fi[s]["op_per_class"][op]["n_gold"] else "–"
                                            for s, _ in SPLITS) + " |")
    rows, excluded = baseline_rows()
    if rows:
        draw_comparison(rows)
        draw_scatter(rows)
        laya_ms = next(r["lat"]["median"] for r in rows if r["name"] == "Laya-Android")
        T += ["", "### Same-input comparison (1,000 test steps)", "",
              "| Model | Params | Serving | Type | Grounding | Median latency | p95 | vs Laya-Android |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            b = "**" if r["name"] == "Laya-Android" else ""
            T.append("| %s%s%s | %s | %s | %s%.1f%s | %s%.1f%s | %s%.0f ms%s | %.0f ms | %s |" % (
                b, r["name"], b, r["params"], r["serving"], b, r["type"], b, b, r["grounding"], b, b,
                r["lat"]["median"], b, r["lat"]["p95"],
                "1x" if r["name"] == "Laya-Android" else "%.1fx slower" % (r["lat"]["median"] / laya_ms)))
        jev = next((r for r in rows if r["serving"] == "TypeSafe API"), None)
        if jev:
            lc = jev["raw"]["laya_android_same_steps"]
            T += ["", "### Calibration vs Jev (same 1,000 steps, operation question)", "",
                  "| Model | ECE (lower is better) | Brier (lower is better) |", "|---|---:|---:|",
                  "| **Laya-Android** | **%.3f** | **%.3f** |" % (lc["ece_operation"], lc["brier_operation"]),
                  "| %s | %.3f | %.3f |" % (jev["name"], jev["raw"]["ece_operation"], jev["raw"]["brier_operation"])]
        if excluded:
            T += ["", "### Excluded from the comparison", "", "| Model | Reason |", "|---|---|"]
            T += ["| %s | %s |" % e for e in excluded]
    with open(os.path.join(R, "tables.md"), "w", encoding="utf8") as f:
        f.write("\n".join(T) + "\n")
    import re
    sync_docs({m.group(1).strip(): m.group(2).strip()
               for m in re.finditer(r"^### (.+?)\n\n(\|.*?)(?=\n\n|\Z)", "\n".join(T), re.S | re.M)})
    print("\n".join(T))


if __name__ == "__main__":
    main()
