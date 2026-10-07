# results/

Everything the README and docs report is generated from these files (`scripts/make_figures.py` rewrites the tables in place).

| path | content | produced by |
|---|---|---|
| `test/final_metrics.json`, `test/base_metrics.json` | AndroidControl-High Type / Grounding (InfiGUI-R1 judge) + internal metrics, fine-tuned and base | `evaluate_androidcontrol.py` |
| `test/final_predictions.jsonl.gz`, `test/base_predictions.jsonl.gz` | per-step actions in the reference format | `evaluate_androidcontrol.py` |
| `test/internal_*.json`, `test/raw/*/` | one-shot test pass: internal metrics and per-step probabilities | `eval_laya_android.py`, `eval_base_laya.py` |
| `val/` | validation: per-epoch training history, final checkpoint, base and smoke runs, predictions, examples | `train_laya_android.py`, `eval_laya_android.py`, `pick_examples.py`, `make_replay.py` |
| `latency/rtx4090.json` | Laya-Android latency profile (candidates, input length, batch size) | `benchmark_latency.py` |
| `baselines/` | same-input comparison on a fixed 1,000-step test subset: generative LLMs (vLLM bf16, Ollama GGUF) and TypeSafe Jev | `bench_vllm.py`, `bench_ollama.py`, `bench_jev.py`, `bench_llm_baselines.py --laya` |
| `external_baselines.json` | numbers reported by other papers, with sources | hand-entered, cited |
| `data/androidcontrol_stats.json` | dataset statistics | `inspect_android_control.py` |
| `dev/smoke_eval_incl_test.json` | development record, see below | `eval_laya_android.py` |
| `tables.md` | all generated tables | `make_figures.py` |

`dev/smoke_eval_incl_test.json`: the 15k-step smoke checkpoint was evaluated on the test subsplits once, before the rule "test only after the configuration is frozen" was adopted. Those numbers were not used for any later decision; they are kept for transparency.
