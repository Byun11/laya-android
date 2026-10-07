---
license: apache-2.0
base_model: convaiinnovations/laya
language:
  - en
tags:
  - android
  - gui-agent
  - accessibility
  - decision-model
  - laya
  - androidcontrol
---

# Laya-Android

> **Status: preliminary.** Training and evaluation are in progress. Final weights and benchmark results are not released yet. Numbers below are validation-only.

## Model Description

Laya-Android is a 322M non-generative decision model specialized for Android GUI action selection from accessibility-based interaction trajectories.

Given a high-level goal, the current accessibility tree (as a list of actionable UI candidates) and recent action history, it predicts the next operation and, for taps, the target UI element. It does this in a single forward pass, without autoregressive text generation and without screenshots.

It is a fine-tune of [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations. The architecture is Laya's, unchanged; Laya-Android contributes the Android data pipeline, candidate construction and the fine-tuned weights. It is not a vision model, a VLM or a world model.

## Model Details

| | |
|---|---|
| Developed by | Jaeyeon Byun (GitHub: Byun11) |
| Model type | Non-generative typed-decision model (bidirectional encoder + option-marker decision head) |
| Parameters | ~322M |
| Encoder | mmBERT-base |
| License | Apache-2.0 |
| Code | https://github.com/Byun11/laya-android |
| Sequence budget | `max_len` 3072, `head_max_len` 1536 |

## Base Model

- `convaiinnovations/laya`, subfolder `multilingual`
- revision `7b928d828b7b0e022f929d9bd2e44165aa270148`
- `laya` Python package 0.3.28
- Laya is Apache-2.0; mmBERT-base is MIT. Laya-Android is not affiliated with or endorsed by Convai Innovations.

## Intended Use

- research on lightweight Android GUI action-selection policies
- accessibility-based mobile agents
- fast System-1 style decision models, for example as a step selector under a planner

## Out-of-Scope Use

- Operating real devices or accounts unsupervised, especially for payments, messaging, account or security settings, or other irreversible actions.
- Tasks that need visual understanding of the screen (images, canvases, games, custom-drawn UI).
- Generating free text: the model does not produce `INPUT_TEXT` content or `OPEN_APP` names.
- Any claim of end-to-end task success: v0 is evaluated offline, step by step.

## Training Data

[AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) (Google Research, Apache-2.0) only, with the official splits.

- train: 13,594 episodes, 74,714 raw actions (9 zero-action episodes excluded), 88,308 decision steps including a synthetic `DONE` after each episode, 132,448 training items (operation and target questions).
- Click grounding coverage on train: 94.76%. Candidates at groundable clicks: median 15, max 246; none truncated.
- Screenshots and low-level step instructions are not used as model input.

## Training Procedure

- Supervised fine-tuning (behavior cloning), cross-entropy over options, hard labels.
- bf16, gradient checkpointing, effective batch 32, encoder lr 2.5e-5, head lr 1e-4, cosine schedule.
- Option order shuffled during training.
- No online RL, no oversampling, no class weighting, no candidate truncation.
- Up to 3 epochs; the checkpoint is selected by validation joint accuracy only, and temperatures are fit on validation only.
- Hardware: 1× RTX 4090 24GB, about 55 minutes per epoch.

## Input Format

Each decision step is two Laya choice questions over one shared state:

```json
{
  "state": {
    "goal": "<high-level goal>",
    "history": ["<last 3 actions>"],
    "ui": ["[0] <label> (<class>) <flags>", "[1] ...", "..."]
  },
  "questions": {
    "operation": {"type": "choice", "instructions": "Which operation should be performed next?",
                  "criteria": {"CLICK": "tap a UI element", "...": "..."}},
    "target": {"type": "choice", "instructions": "Which UI element should be targeted?",
               "criteria": {"0": "<label> (<class>) <flags>", "1": "..."}}
  }
}
```

Use the repository's `src/candidates.py` and `src/serialization.py` to build exactly this input from an accessibility forest; other serializations are out of distribution.

## Output Format

A probability distribution over options for each question:

- `operation`: one of `CLICK, LONG_PRESS, SCROLL_UP, SCROLL_DOWN, SCROLL_LEFT, SCROLL_RIGHT, OPEN_APP, INPUT_TEXT, BACK, HOME, WAIT, DONE`
- `target`: the index of a UI candidate (asked for `CLICK` / `LONG_PRESS`)

## Evaluation

Offline, step-level, on AndroidControl, with the high-level goal only. Metrics: operation accuracy, operation macro-F1 with per-class precision/recall, target top-1 on groundable clicks, joint accuracy, grounding coverage, ECE and Brier, target top-1 by candidate count, and latency. Text payloads are not scored, so the numbers are not directly comparable to AndroidControl step accuracy reported elsewhere.

### Preliminary (validation only)

| AndroidControl validation (827 steps) | Base Laya (zero-shot) | Laya-Android, epoch 1 |
|---|---:|---:|
| joint accuracy | 0.098 | 0.605 |
| target top-1 | 0.110 | 0.631 |
| operation accuracy | 0.223 | 0.768 |
| operation macro-F1 | 0.142 | 0.543 |
| operation ECE | 0.604 | 0.027 |

These are validation results, not held-out final test results. Test results will be added once the checkpoint is final.

## Limitations

- No visual input.
- Relies on accessibility semantics; unlabeled icons are hard to distinguish.
- Arbitrary text generation is unsupported.
- Weak on actions with little supervision (for example `LONG_PRESS`, 157 training examples).
- Target performance is bounded by candidate quality; about 5% of clicks cannot be grounded.
- Custom-drawn or inaccessible UI may be unobservable.
- v0 is AndroidControl-specialized and has not been evaluated online.

## Ethical / Safety Considerations

- A GUI action model can trigger real side effects on a device. Keep a human in the loop and gate sensitive or irreversible actions.
- Accessibility trees can contain personal data shown on screen (messages, contacts, account details). Process them locally and do not log them unnecessarily.
- Calibrated confidence (ECE above) can support abstention, but low confidence does not guarantee safety and high confidence does not guarantee correctness.
- The training data comes from crowd-sourced demonstrations on a limited set of apps and may not reflect other apps, locales or accessibility needs.

## Citation

```bibtex
@software{laya_android_2026,
  title   = {Laya-Android},
  author  = {Byun, Jaeyeon},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Byun11/laya-android}
}

@article{li2024effects,
  title={On the Effects of Data Scale on Computer Control Agents},
  author={Li, Wei and Bishop, William and Li, Alice and Rawles, Chris and Campbell-Ajala, Folawiyo and Tyamagundlu, Divya and Riva, Oriana},
  journal={arXiv preprint arXiv:2406.03679},
  year={2024}
}
```

## Acknowledgements

- Built on top of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache-2.0).
- Encoder: [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT).
- Data: [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) by Google Research (Apache-2.0).
