---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: convaiinnovations/laya
base_model_relation: finetune
language:
  - en
tags:
  - laya
  - system-one
  - decision-model
  - android
  - mobile-agent
  - gui-agent
  - accessibility
  - androidcontrol
  - mmbert
model-index:
  - name: laya-android
    results:
      - task:
          type: other
          name: Android GUI action selection
        dataset:
          name: AndroidControl-High (InfiGUI-R1 evaluator, 8,444 test steps)
          type: InfiX-ai/android_control_test
          split: test
        metrics:
          - type: accuracy
            name: Type
            value: 76.7
          - type: accuracy
            name: Grounding
            value: 61.7
---

# Laya-Android

A 322M typed-decision policy for Android GUI control, fine-tuned from [Laya](https://huggingface.co/convaiinnovations/laya) on AndroidControl. Accessibility tree only · no vision · no autoregressive generation.

[Code, docs and full results on GitHub](https://github.com/Byun11/laya-android)

## Highlights

| Parameters | AndroidControl-High Type | AndroidControl-High Grounding | Latency (RTX 4090, bs1) |
|:---:|:---:|:---:|:---:|
| **322M** | **76.7** | **61.7** | **~40 ms** |

![Laya-Android on a validation episode](figures/replay_success.png)

Every step of a recorded validation episode with the model's predictions (blue = predicted, green = gold). The model never sees the screenshots; it reads the accessibility tree. Gold history, not a live run.

## Usage

```python
# pip install laya==0.3.28
import laya

agent = laya.load("ByunByun/laya-android")

OP_DESC = {"CLICK": "tap a UI element", "LONG_PRESS": "long-press a UI element", "SCROLL_UP": "scroll up",
           "SCROLL_DOWN": "scroll down", "SCROLL_LEFT": "scroll left", "SCROLL_RIGHT": "scroll right",
           "OPEN_APP": "launch an app by name", "INPUT_TEXT": "type text into the focused field",
           "BACK": "press the system back button", "HOME": "go to the home screen",
           "WAIT": "wait for the screen to update", "DONE": "the goal is complete; stop"}

goal = "Turn off Bluetooth"
history = ["OPEN_APP Settings"]                      # last 3 actions
candidates = ["Network & internet (LinearLayout)", "Connected devices (LinearLayout)", "Apps (LinearLayout)"]

state = {"goal": goal, "history": history, "ui": ["[%d] %s" % (i, c) for i, c in enumerate(candidates)]}
questions = {
    "operation": {"type": "choice", "instructions": "Which operation should be performed next?", "criteria": OP_DESC},
    "target": {"type": "choice", "instructions": "Which UI element should be targeted?",
               "criteria": {str(i): c for i, c in enumerate(candidates)}},
}
out = agent.system_one(state, questions, max_len=3072, head_max_len=1536)
print(out["answers"]["operation"]["choice"], out["answers"]["target"]["choice"])
```

Candidate labels must follow the repository's `src/candidates.py` / `src/serialization.py` (`"<text> (<class>) <flags>"`).

## Evaluation

AndroidControl-High, 8,444 test steps, public InfiGUI-R1 evaluator. Same model before and after fine-tuning:

<!-- table: AndroidControl-High: base vs fine-tuned (official evaluator) -->
| Model | Params | Type | Grounding | Full SR |
|---|---:|---:|---:|---:|
| Laya base (zero-shot) | 322M | 29.0 | 11.8 | N/A |
| **Laya-Android** | **322M** | **76.7** | **61.7** | **N/A** |
| Δ | | +47.6 | +49.9 | |
<!-- /table -->

Full SR is N/A (v0 does not generate text or app names). Screenshot VLMs on the same evaluator report higher numbers (e.g. InfiGUI-R1-3B 82.7 / 74.4) but solve a different task (pixel coordinates vs. choosing accessibility candidates).

<!-- table: Base Laya → Laya-Android (Policy Joint Accuracy, test) -->
| Split | Base Laya | Laya-Android | Δ |
|---|---:|---:|---:|
| IDD | 0.115 | **0.659** | +54.4pt |
| App-Unseen | 0.112 | **0.540** | +42.8pt |
| Task-Unseen | 0.108 | **0.567** | +45.9pt |
| Category-Unseen | 0.108 | **0.545** | +43.7pt |
<!-- /table -->

Policy Joint = operation correct and, for clicks, the gold element chosen (internal metric, not SR). One test pass of the frozen checkpoint; protocol fixed before scoring ([eval_protocol.md](https://github.com/Byun11/laya-android/blob/main/docs/eval_protocol.md)).

## Compared with generative models

![Accuracy and latency, same input](figures/efficiency_scatter.png)

<!-- table: Same-input comparison (1,000 test steps) -->
| Model | Params | Serving | Type | Grounding | Median latency | p95 | vs Laya-Android |
|---|---|---|---:|---:|---:|---:|---:|
| **Laya-Android** | 322M | Laya-Android (local) | **77.2** | **62.8** | **39 ms** | 48 ms | 1x |
| Qwen3.8 27B | 27B (Q4_K_M) | Ollama GGUF | 76.8 | 64.5 | 575 ms | 694 ms | 14.9x slower |
| jev-1.13.0 | undisclosed | TypeSafe API | 67.6 | 56.4 | 244 ms | 308 ms | 6.3x slower |
| Qwen3.5 4.2B | 4.2B (Q4_K_M) | Ollama GGUF | 67.3 | 52.8 | 193 ms | 252 ms | 5.0x slower |
| Qwen3.5 4B | 4B (bf16) | vLLM bf16 | 67.1 | 50.5 | 242 ms | 305 ms | 6.3x slower |
| Gemma 4 25B | 25B (Q4_K_M) | Ollama GGUF | 64.8 | 56.6 | 201 ms | 256 ms | 5.2x slower |
| Qwen3.5 9.7B | 9.7B (Q4_K_M) | Ollama GGUF | 63.6 | 51.6 | 229 ms | 273 ms | 5.9x slower |
| Gemma 4 E2B | E2B (bf16) | vLLM bf16 | 58.4 | 23.6 | 155 ms | 210 ms | 4.0x slower |
| Qwen3.5 2B | 2B (bf16) | vLLM bf16 | 57.5 | 23.0 | 149 ms | 202 ms | 3.9x slower |
| Gemma 4 8.0B | 8.0B (Q4_K_M) | Ollama GGUF | 57.4 | 45.0 | 171 ms | 202 ms | 4.4x slower |
| Qwen3.5 1.9B | 1.9B (Q8_0) | Ollama GGUF | 57.4 | 24.5 | 126 ms | 144 ms | 3.3x slower |
| Gemma 4 E4B | 7.5B (Q4_K_M) | Ollama GGUF | 57.1 | 44.1 | 146 ms | 175 ms | 3.8x slower |
| Gemma 4 E2B | 4.6B (Q4_K_M) | Ollama GGUF | 56.9 | 22.7 | 116 ms | 139 ms | 3.0x slower |
| Ministral 3 14B | 14B (Q4_K_M) | Ollama GGUF | 53.2 | 42.0 | 285 ms | 424 ms | 7.4x slower |
| Qwen3.5 0.8B | 0.8B (Q8_0) | Ollama GGUF | 41.3 | 0.0 | 124 ms | 158 ms | 3.2x slower |
| Qwen3.5 0.8B | 0.8B (bf16) | vLLM bf16 | 40.9 | 0.4 | 125 ms | 178 ms | 3.2x slower |
<!-- /table -->

Same 1,000 test steps (seed 0), same input (goal, last 3 actions, the same accessibility candidates), same InfiGUI-R1 judge, one RTX 4090 at batch 1. The generative models are used zero-shot (not trained on AndroidControl), thinking disabled, asked for a JSON action; vLLM runs them in bf16, Ollama in 4-bit GGUF. Jev is TypeSafe's hosted decision model (jev-1.13.0, measured 2026-10-07) and its latency is an API round trip. The comparison shows the speed of answering by selection versus generation; the accuracy gap also reflects that only Laya-Android was trained on this dataset.

## Model details

- Base: `convaiinnovations/laya` / `multilingual` @ `7b928d8` (mmBERT-base encoder), ~322M parameters, Apache-2.0
- Input: goal, last 3 actions, accessibility-derived UI candidates; `max_len` 3072, `head_max_len` 1536
- Output: probabilities over 12 operations and over the candidates
- Developed by Jaeyeon Byun (GitHub: Byun11); not affiliated with Convai Innovations

## Training

AndroidControl train split only (13,594 episodes, 74,714 actions). Supervised fine-tuning with `laya.train`, bf16, 3 epochs on one RTX 4090; checkpoint and temperatures chosen on validation. No RL, no oversampling, no screenshots. Details: [training.md](https://github.com/Byun11/laya-android/blob/main/docs/training.md).

## Limitations

- No visual input: unlabeled icons and custom-drawn UI are hard to tell apart.
- No text or app-name generation, so no Full Step SR; AndroidControl-Low not evaluated.
- Policy Joint drops 9–12 points on unseen apps, tasks and categories.
- Weak on rare actions (WAIT, BACK, horizontal scroll, LONG_PRESS).
- Offline, step-level benchmark with gold history. Do not let it operate real devices unsupervised; gate irreversible actions and keep on-screen personal data local.

## Citation

```bibtex
@software{laya_android_2026,
  title   = {Laya-Android},
  author  = {Byun, Jaeyeon},
  year    = {2026},
  version = {0.2.1},
  url     = {https://github.com/Byun11/laya-android}
}
```

Built on [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations), [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) and [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) (Google Research). Evaluator: [InfiGUI-R1](https://github.com/InfiXAI/InfiGUI-R1).
