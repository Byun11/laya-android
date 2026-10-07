"""Zero-shot base Laya on the same harness: python scripts/eval_base_laya.py --out reports/base_laya_eval.json"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_laya_android import main  # noqa: E402

if __name__ == "__main__":
    main(default_model="multilingual")  # convaiinnovations/laya@7b928d8, subfolder multilingual
