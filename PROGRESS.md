# PROGRESS — Laya-Android v0

## 2026-10-07

### 1. Repository / environment
- Machine: RTX 4090 24GB, Windows 11, Python 3.10.5. No A100 locally.
- Global env is unusable as-is: torch 2.13.0 is a **CPU-only** build; TF 2.15 fails to import (numpy 2.2 / protobuf 6.33).
- Dedicated venv `D:/laya-android/.venv` (`--system-site-packages`) with `laya==0.3.28` and `torch 2.14.1+cu130` (bf16 OK).
- Data, checkpoints and venv live on `D:/laya-android` (code folder is OneDrive-synced; 50GB must not sync). Override with `LAYA_ANDROID_DATA`.
- Base model: `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, subfolder `multilingual` (mmBERT-base, 322M). Published config max_len 1024 / head_max_len 256; we run 3072 / 1536 (mmBERT supports 8192).

### 2–4. Download, TFRecord parser, split loader
- GCS `gresearch/android_control`: 20 GZIP TFRecord shards (49.9GB) + `splits.json` + `test_subsplits.json`.
- Pure-Python wire decoder (`src/data/android_control.py`); screenshots skipped.
- Splits: train 13,603 / validation 137 / test 1,543 episodes; subsplits IDD 721, app_unseen 631, task_unseen 803, category_unseen 700 (overlapping, all ⊂ test).
- Found: some episodes have **0 actions** (feature absent) → skipped and counted (1 in shard 0).
- Accessibility bools are serialized explicitly (false present), screens up to ~3.5k nodes.

### 6–8. Candidates and grounding (`src/candidates.py`)
- Candidate = visible, non-IME node with any of clickable/long_clickable/editable/scrollable/checkable; box clipped to screen.
- Label = own text/content-desc/hint, else descendant texts (≤60 chars).
- Dedup on (box, label), keep deepest node, OR the flags. Order: (top, left, bottom, right, label, class).
- Grounding: smallest containing candidate → deepest → lowest index; none → ungroundable.
- Shard-0 numbers (train part, 699 episodes): grounding coverage **95.2%**; candidates/screen p50 16, p90 32, p99 60, max 246; gold has empty label 9.6%; gold is scroll-container-only 2.1%.
- Untruncated target head tokens p99 827 / max 1357; state tokens p99 778 / max 1421 → max_len 3072, head_max_len 1536 keeps every candidate.

### 11. Base Laya (trial, 200 test_idd steps, not the final number)
- op 0.325, target top-1 0.136, joint 0.132, latency median 39ms (batch 1, both questions, bf16 autocast).

### 12. Training pipeline
- `laya.train.train_model` reused; `_forward` monkeypatched to bf16; best epoch by val joint; temperatures fit on val.
- Tiny run (406 steps) end-to-end OK: ~35 items/s, 2.5GB VRAM at micro_batch 8 with grad checkpointing.

### 9–10. Full dataset (`reports/androidcontrol_stats.json`)
- 9 episodes with 0 actions skipped. Train 13,594 episodes / 74,714 actions (+13,594 DONE = 88,308 steps); paper: 13,604 / 74,722.
- Steps: val 827, test_idd 4,618, app_unseen 4,106, task_unseen 5,267, category_unseen 4,591 (incl. DONE).
- Train ops: CLICK 46,424, SCROLL_DOWN 7,721, INPUT_TEXT 5,406, WAIT 5,159, OPEN_APP 5,044, BACK 2,662, SCROLL_UP 1,043, SCROLL_RIGHT 861, SCROLL_LEFT 208, LONG_PRESS 157, HOME 29.
- Grounding coverage: train 94.8%, val 95.3%, IDD 95.2%, app-unseen 93.3%, task-unseen 92.3%, category-unseen 93.8%.
- Candidates at groundable clicks (train): p50 15, p90 33, p99 65, max 246; buckets ≤5 10.7%, 6–10 18.1%, 11–20 39.8%, 21–50 28.8%, >50 2.6%.
- Tokens (3k sampled train steps): state p99 769 / max 1,635; untruncated target head p99 806 / max 1,641. **0 unfit items** in every eval split at 3072/1536.

### 11. Base Laya zero-shot (`reports/base_laya_eval.json`, high-level goal only)
| split | op | target top-1 | joint | coverage | ECE op | latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.223 | 0.110 | 0.098 | 0.953 | 0.604 | 38.7ms |
| IDD | 0.274 | 0.122 | 0.115 | 0.952 | 0.547 | 38.9ms |
| app-unseen | 0.250 | 0.143 | 0.112 | 0.933 | 0.572 | 38.6ms |
| task-unseen | 0.233 | 0.144 | 0.108 | 0.923 | 0.589 | 38.5ms |
| category-unseen | 0.241 | 0.133 | 0.108 | 0.938 | 0.581 | 38.6ms |
- Target top-1 by candidates (IDD): 1–5 0.335, 6–10 0.150, 11–20 0.107, 21–50 0.056, >50 0.029.

### 12. Smoke SFT (`configs/smoke.yaml`, `reports/train_smoke.json`)
- 15,003 steps (whole episodes, seed 0) → 22,474 items, 0 skipped. 1 epoch, 0.156h on RTX 4090.
- Loss (rerun with windowed logging, mean per 500 micro-batches): 1.617 → 1.365 → 1.261 → 1.201 → 1.189. Decreasing.
- Run-to-run noise: identical config rerun gave val joint 0.444 vs 0.456 (GPU nondeterminism) → treat ~1pt as noise.

### 13. Smoke eval (`reports/smoke_eval.json`)
| split | op | target top-1 | joint | ECE op |
|---|---:|---:|---:|---:|
| val | 0.689 | 0.476 | 0.457 | 0.038 |
| IDD | 0.692 | 0.472 | 0.445 | 0.025 |
| app-unseen | 0.644 | 0.529 | 0.450 | 0.043 |
| task-unseen | 0.663 | 0.535 | 0.468 | 0.035 |
| category-unseen | 0.650 | 0.532 | 0.457 | 0.037 |
- Target by candidates (IDD) base→smoke: 1–5 .335→.748, 6–10 .150→.480, 11–20 .107→.463, 21–50 .056→.391, >50 .029→.353.
- Weak ops (IDD recall): BACK 0.05, WAIT 0.16, DONE 0.38, SCROLL_LEFT/RIGHT ~0.12, LONG_PRESS 0 (157 train examples).
- Stop/Go: all five checks pass → GO for full, pending user decision.

### Full run v0 (`configs/full.yaml`, `reports/train_laya-android-v0.json`, `reports/full_v0_val.json`)
- 88,308 steps → 132,448 items, 0 skipped. 3 epochs, 2.726h on RTX 4090. All epoch checkpoints kept (`epoch1..3`).
- Validation only (test not run):

| epoch | train loss | op acc | op macro-F1 | target top-1 | joint | ECE op (uncalibrated) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.019 | 0.768 | 0.543 | 0.631 | 0.605 | 0.027 |
| 2 | 0.653 | 0.797 | 0.542 | 0.703 | **0.669** | 0.028 |
| 3 | 0.366 | 0.790 | 0.574 | 0.703 | 0.665 | 0.105 |

- Selected: **epoch 2** (val joint). ep2 vs ep3 gap 0.004 is inside run-to-run noise (~1pt); rule applied as written. Epoch 3 shows overfitting signs: loss drops in steps at epoch boundaries, uncalibrated ECE 0.105.
- Final (`checkpoints/laya-android-v0`, val temperatures op 1.161): val joint 0.669, target 0.703, op 0.797, macro-F1 0.542, ECE op 0.027 (in-sample: fit on the same val), latency p50 35ms.
- Target by candidates (val, ep2): 1–5 0.842, 6–10 0.711, 11–20 0.695, 21–50 0.634, >50 0.667 (n=6).
- Op recall (val, ep2): CLICK .881, OPEN_APP .956, INPUT_TEXT .849, DONE .759, BACK .655, SCROLL_UP .667, SCROLL_DOWN .634, WAIT .482, LONG_PRESS 0, SCROLL_LEFT 0, SCROLL_RIGHT 0 (tiny val n).
