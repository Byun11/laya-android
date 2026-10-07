# Laya-Android

Laya-Android is a 322M non-generative decision model specialized for Android GUI action selection from accessibility-based interaction trajectories.

It adapts the open [Laya](https://github.com/NandhaKishorM/laya) typed-decision model by Convai Innovations to Android interaction trajectories from [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control).

## Key idea

Given:
- a high-level user goal,
- the current Android accessibility tree,
- recent action history,
- and a finite set of legal UI candidates,

Laya-Android predicts:
1. the next operation, and
2. the target UI element when applicable,

without autoregressive text generation or screenshot encoding.

## Status

Training and evaluation are in progress.

The first release is trained only on AndroidControl.
Weights and final benchmark results will be released after evaluation is complete.

---

## Architecture

Laya-Android does not propose a new architecture. It uses Laya's non-generative typed-decision architecture unchanged and fine-tunes its weights.

```text
Goal + UI state + history + options
                 ↓
      bidirectional encoder          (mmBERT-base inside the Laya multilingual checkpoint)
                 ↓
      option marker states           (one [MASK] marker per option)
                 ↓
         decision head               (Laya head: 2 transformer layers + scorer)
                 ↓
       probability / option
```

Every operation and every UI candidate is a request-time option. A decision is one forward pass over one sequence; nothing is decoded token by token.

What this project adds is the Android adaptation:

```text
Android accessibility tree
        ↓
actionable UI candidates            (src/candidates.py)
        ↓
Laya typed decision format          (src/serialization.py)
        ↓
operation + target selection
```

| | |
|---|---|
| Base model | `convaiinnovations/laya`, subfolder `multilingual`, revision `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Encoder | mmBERT-base |
| Parameters | ~322M (encoder + Laya decision head) |
| Laya package | `laya==0.3.28` |
| Sequence budget | `max_len` 3072, `head_max_len` 1536 |

## Training Data

AndroidControl only (official `splits.json` / `test_subsplits.json`). Screenshots exist in the dataset but are **not** used as model input.

| | episodes | raw actions | decision steps (incl. synthetic DONE) |
|---|---:|---:|---:|
| train | 13,594 | 74,714 | 88,308 |
| validation | 137 | 690 | 827 |

- 9 episodes with zero actions are excluded.
- The four official test subsplits (IDD, App-Unseen, Task-Unseen, Category-Unseen) overlap by design and are written to separate files.
- One decision step yields one or two training items (an operation question and, for groundable clicks, a target question): 132,448 items for the full train split.

Dataset statistics are in [`reports/androidcontrol_stats.json`](reports/androidcontrol_stats.json), which is the source of truth for the numbers below.

## Input Representation

Main setting: the high-level goal only. The per-step low-level instruction is **not** in the main input; it is available only as an `--low-level` oracle diagnostic.

```json
{
  "goal": "Turn off Bluetooth",
  "history": ["OPEN_APP Settings"],
  "ui": ["[0] Network & internet (LinearLayout)", "[1] Connected devices (LinearLayout)", "..."]
}
```

- `history`: the last 3 actions (`CLICK <label>`, `INPUT_TEXT "<text>"`, `OPEN_APP <name>`, or the bare operation).
- `ui`: every candidate as `[index] label (class) flags`. Flags: `edit`, `long`, `scroll`, `on`, `selected`, `disabled`.
- The same candidate labels are the options of the target question.

## Action Space

| AndroidControl action | Laya-Android operation |
|---|---|
| `click` | `CLICK` (+ target) |
| `long_press` | `LONG_PRESS` (+ target) |
| `scroll` + direction | `SCROLL_UP` / `SCROLL_DOWN` / `SCROLL_LEFT` / `SCROLL_RIGHT` |
| `open_app` | `OPEN_APP` (app name not predicted) |
| `input_text` | `INPUT_TEXT` (text not predicted) |
| `navigate_back` | `BACK` |
| `navigate_home` | `HOME` |
| `wait` | `WAIT` |
| — | `DONE` (synthetic, appended after the last observation of each episode) |

## Candidate Grounding

Candidates (`src/candidates.py`) are a pure function of the accessibility forest; the gold action is never used to build them.

1. Keep visible nodes outside the input-method (keyboard) window that are clickable, long-clickable, editable, scrollable or checkable. Clip boxes to the screen.
2. Label = own text / content description / hint, else the descendant texts (≤ 60 chars).
3. Deduplicate on (box, label), keeping the deepest node and OR-ing its action flags.
4. Order by (top, left, bottom, right, label, class).

A gold `click(x, y)` / `long_press(x, y)` maps to the smallest candidate box containing the point, then the deepest node, then the lowest index. If no candidate contains the point, the step is ungroundable: it stays in the history, has no target question, and is excluded from target and joint metrics.

| split | click grounding coverage |
|---|---:|
| train | 94.76% (44,140 / 46,581) |
| validation | 95.25% |

Candidates at groundable train clicks: median 15, p90 33, p99 65, max 246. No candidate is truncated: at `head_max_len` 1536 / `max_len` 3072, no evaluation item fails to fit.

## Training

Supervised fine-tuning (behavior cloning) with `laya.train.train_model`:

- loss: cross-entropy over options (`soft-ce` with hard labels)
- bf16 autocast (Laya 0.3.28 hardcodes fp16; replaced in `scripts/train_laya_android.py`)
- option order shuffled per item during training; canonical order at evaluation
- effective batch 32 questions; encoder lr 2.5e-5, head lr 1e-4, cosine schedule
- gradient checkpointing
- no online RL, no oversampling, no class weighting, no candidate truncation
- checkpoint selection by validation joint accuracy only; temperatures fit on validation only

Configs: [`configs/smoke.yaml`](configs/smoke.yaml) (15k steps, 1 epoch) and [`configs/full.yaml`](configs/full.yaml) (full train split, 3 epochs). Hardware: a single RTX 4090 (24GB); one full epoch takes about 55 minutes.

## Evaluation

`scripts/eval_laya_android.py` evaluates base and fine-tuned checkpoints with the same harness:

- **operation accuracy** over all steps, including DONE
- **operation macro-F1**, with per-class precision / recall / F1
- **target top-1** over groundable CLICK / LONG_PRESS steps
- **joint accuracy**: operation correct and, for click steps, target correct (ungroundable click steps excluded)
- **grounding coverage**, **ECE** (15 bins) and **Brier** score for both questions
- target top-1 by candidate count (1–5, 6–10, 11–20, 21–50, >50)
- latency: one forward pass with both questions at batch size 1

Text payloads (`INPUT_TEXT` text, `OPEN_APP` app name) are not scored, so these numbers are not directly comparable to AndroidControl step accuracy in the literature.

## Current Results

### Preliminary results

Results below are validation-only and are not final benchmark numbers.

AndroidControl validation (827 steps), high-level goal only:

| | Base Laya (zero-shot) | Laya-Android, full run epoch 1 |
|---|---:|---:|
| joint accuracy | 0.098 | 0.605 |
| target top-1 | 0.110 | 0.631 |
| operation accuracy | 0.223 | 0.768 |
| operation macro-F1 | 0.142 | 0.543 |
| operation ECE | 0.604 | 0.027 |

These are validation results, not held-out final test results. The final checkpoint has not been selected yet.

## Limitations

- No visual input: the model sees only accessibility-tree text.
- It relies on accessibility semantics. Custom-drawn or inaccessible UI may be unobservable, and unlabeled icons (about 9% of gold click targets in train) are hard to tell apart.
- It cannot generate text. `INPUT_TEXT` content and `OPEN_APP` names need a separate component.
- Actions with little supervision are weak (for example `LONG_PRESS`, 157 training examples).
- Target accuracy depends on candidate quality; about 5% of clicks cannot be grounded to any candidate.
- v0 is specialized to AndroidControl and has not yet been evaluated online (for example in AndroidWorld).

## Reproduction

```bash
pip install -r requirements.txt
export LAYA_ANDROID_DATA=/path/to/large/disk      # default D:/laya-android

# 1. download AndroidControl (~50GB) into $LAYA_ANDROID_DATA/raw/android_control
B=https://storage.googleapis.com/gresearch/android_control
curl -fLO $B/splits.json && curl -fLO $B/test_subsplits.json
for i in $(seq -w 0 19); do curl -fLO $B/android_control-000$i-of-00020; done

# 2. parse + build step-level JSONL, then statistics
python scripts/build_androidcontrol_dataset.py --workers 10
python scripts/inspect_android_control.py

# 3. zero-shot base, train, evaluate on validation
python scripts/eval_base_laya.py --splits val --out reports/base_laya_val.json
python scripts/train_laya_android.py configs/full.yaml
python scripts/eval_laya_android.py --model $LAYA_ANDROID_DATA/checkpoints/laya-android-v0 --splits val --out reports/val.json
```

The parser is pure Python and needs neither TensorFlow nor android_env.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite AndroidControl:

```bibtex
@article{li2024effects,
  title={On the Effects of Data Scale on Computer Control Agents},
  author={Li, Wei and Bishop, William and Li, Alice and Rawles, Chris and Campbell-Ajala, Folawiyo and Tyamagundlu, Divya and Riva, Oriana},
  journal={arXiv preprint arXiv:2406.03679},
  year={2024}
}
```

## Acknowledgements

- Built on top of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache-2.0). Laya-Android is not affiliated with or endorsed by Convai Innovations.
- Encoder: [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT).
- Data: [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) by Google Research (Apache-2.0).

Licensed under Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
