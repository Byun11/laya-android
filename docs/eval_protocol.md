# Evaluation protocol (frozen before computing Type / Grounding)

This file fixes every metric definition **before** the community-comparable metrics are computed from the saved
test predictions. It must not change after those numbers are seen. Result JSONs record the git commit of this file.

## 0. What is frozen

| item | value |
|---|---|
| checkpoint | `laya-android-v0` = epoch 2 of `configs/full.yaml` (selected by validation Policy Joint only) |
| temperature | fit on validation before any test evaluation; not refit on test |
| test inference | run **once** (2026-10-07) on the four official subsplits; per-step probabilities saved |
| baseline | `convaiinnovations/laya@7b928d8` / `multilingual`, same pass, same harness, same inputs |
| input | high-level goal + accessibility-derived candidates + last 3 gold actions (no screenshot, no step instruction) |

Type / Grounding are computed **offline from the saved predictions**; no further inference on test.

## 1. Reference evaluator

InfiGUI-R1, `eval/android_control/` at commit `a4fca17809a4395ba1fe08d481bb82c790ea7236`
(https://github.com/InfiXAI/InfiGUI-R1), test data `InfiX-ai/android_control_test` (`android_control_test.json`).

The per-step judge `evaluate_android_control_action` (evaluate_android_control.py:80-120) is called **unmodified**;
the aggregation reproduces `compute_scores` (android_control.py:259-278).

## 2. Sample universe

- All 1,543 official test episodes; one sample per action `actions[k]` on observation `k` → **8,444 steps**.
- Join key: `(episode_id, step k)`; matches the reference test file 8,444 / 8,444 with identical action types.
- Our synthetic `DONE` rows (one per episode, after the last action) are **excluded**.
- Our four subsplit files overlap; each `(episode_id, k)` is counted once in the overall numbers. Per-subsplit
  numbers use that subsplit's episodes.
- Ground truth (`step_check_pams`: action, `candidate_bbox`, `coordinate`, `text`, `direction`, `button`) is taken
  from the reference test file, not from our pipeline.

## 3. Converting a Laya-Android prediction to the reference action format

The predicted operation is the argmax of the operation question. Coordinates are original screen pixels; the
evaluator is called with `resized_width = width`, `resized_height = height`.

| our operation | reference action | arguments we can supply |
|---|---|---|
| `CLICK` | `click` | `coordinate` = center of the argmax target candidate's box |
| `LONG_PRESS` | `long_press` | same as CLICK |
| `SCROLL_UP/DOWN/LEFT/RIGHT` | `swipe` | `coordinate`, `coordinate2` = screen center → a point offset in that direction (affects only exact match) |
| `INPUT_TEXT` | `type` | `text` = `""` (v0 does not generate text) |
| `OPEN_APP` | `open` | `text` = `""` (v0 does not generate app names) |
| `BACK` | `system_button` | `button` = `Back` |
| `HOME` | `system_button` | `button` = `Home` |
| `WAIT` | `wait` | – |
| `DONE` | `terminate` | – (no test step has this type, so it is always a type error) |

Target availability: the target question was asked (and saved) only on steps whose gold action is a groundable
CLICK / LONG_PRESS. When a click/long_press must be emitted and no saved target exists (gold not groundable by our
extractor, or the gold action is not a click), the coordinate is `(-1, -1)`, i.e. a guaranteed miss. This is
conservative: it never adds a hit.

## 4. Metrics

**Type** = `type_match` over all 8,444 steps (evaluate_android_control.py:84-120):
action types must match exactly; `click` ≠ `long_press`; swipe direction is ignored; `system_button` matches any
button; a predicted `click` on a gold `open` step counts as a type match iff that step has a non-empty
`candidate_bbox` (reference behavior, kept as is).

**Grounding** = `#(exact_match ∧ predicted action == click) / #(gold action == click)` (android_control.py:265-272).
A click is a hit if the point lies in any gold `candidate_bbox` enlarged 1.2× around its center, or within
`0.04 · max(W, H)` pixels of the gold point (evaluate_android_control.py:6-7, 15-26). The predicted type must be
`click`, so a type error is also a grounding miss. Gold `long_press` steps are not in the denominator. Hits on gold
`open` steps via click enter the numerator (reference behavior; with v0 these cannot occur, see §3).

**Full SR (exact match) — not reported.** It requires `type` text and `open` app names. v0 emits empty strings,
and the reference `check_text` treats an empty prediction as a substring of any gold text
(evaluate_android_control.py:9-12), so the number would be inflated and meaningless. Reported as **N/A**.

**Policy Joint (internal, not comparable to SR)** = operation correct ∧ (for gold CLICK / LONG_PRESS: argmax target
candidate == gold candidate). Over our step rows **including** synthetic DONE; gold clicks that our extractor cannot
ground are excluded from its denominator and reported as **candidate coverage**. Scroll direction is part of the
operation. Payloads are not scored.

**Other internal metrics** (same saved predictions): operation accuracy and macro-F1 over our 12 operations
(incl. DONE), per-operation precision / recall / support, target candidate top-1 on groundable clicks, target top-1
by candidate count (1–5, 6–10, 11–20, 21–50, >50, with n), ECE (15 bins, top-label) and Brier with the frozen
temperature.

## 5. Comparability notes (to appear next to every table)

- Laya-Android picks from accessibility-derived candidates instead of predicting free screen coordinates. The gold
  `candidate_bbox` also comes from the accessibility tree, which favors this setup.
- About 5% of gold clicks are not covered by our candidate extractor and are unrecoverable (counted as misses).
- History is given as gold previous actions, as in the reference evaluator.
- External rows are included only if they were produced with this evaluator family (8,444 steps, bbox-or-4%
  click rule). Results from the 7,708-step / 14%-distance protocol are not mixed in.

## 6. AndroidControl-Low

**Not evaluated for v0.** The model was trained only with the high-level goal; step instructions were never part
of its input, so a Low number would mix task ability with input distribution shift.

## 7. Latency (separate, `scripts/benchmark_latency.py`)

RTX 4090 24GB, fp32 weights + bf16 autocast, batch 1, 100 warm-up + 1,000 measured steps from test_idd inputs
(labels unused), operation + target questions in one forward, CUDA-synchronized. Data loading excluded from
`forward_ms`; tokenization included in `e2e_ms`.
