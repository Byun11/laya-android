# reports/

Small JSON outputs of the scripts. All numbers are AndroidControl, high-level goal only, unless noted.

| file | produced by | content |
|---|---|---|
| `androidcontrol_stats.json` | `scripts/inspect_android_control.py` | dataset statistics per split (source of truth for README numbers) |
| `base_laya_val.json` | `scripts/eval_base_laya.py --splits val` | zero-shot base Laya, validation |
| `base_laya_eval.json` | `scripts/eval_base_laya.py` | zero-shot base Laya, validation + 4 test subsplits |
| `train_smoke.json` | `scripts/train_laya_android.py configs/smoke.yaml` | smoke run (15k steps, 1 epoch) log |
| `smoke_val.json` | `scripts/eval_laya_android.py --splits val` | smoke checkpoint, validation |
| `smoke_eval.json` | `scripts/eval_laya_android.py` | smoke checkpoint, validation + 4 test subsplits |

Notes:
- The smoke checkpoint was evaluated on the test subsplits once, before the rule "test only after the configuration is frozen" was adopted. Those smoke test numbers were not used for any later decision.
- Final results (test metrics, predictions, latency, figures) live in `results/` and `figures/`; see `results/tables.md`.
- `model` fields hold the local checkpoint path that was evaluated.
