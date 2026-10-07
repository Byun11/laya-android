"""Laya-Android quick start: one decision step from a goal, recent actions and accessibility-derived UI candidates.

pip install laya==0.3.28 ; run from the repository root: python examples/quickstart.py [model_id_or_path]
Candidate labels must be rendered as in src/serialization.py (`option_label`): "<text> (<class>) <flags>".
"""
import sys

import laya

sys.path.insert(0, "src")
from serialization import OP_DESC, Q_OP, Q_TARGET  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "ByunByun/laya-android"
agent = laya.load(model)

goal = "My old shoes are damaged and I'm planning to buy new casual shoes ,find a nike casual shoes for women on the kicks crew app"
history = ['INPUT_TEXT "casual shoes for women"', "CLICK (WMNS) PUMA Oslo Maja For Casual Shoes White/Purple/Yellow U",
           "CLICK Filter"]  # last 3 actions
candidates = ["Dismiss (View)", "FILTER Clear All Brand Size And Type Rel (View) scroll", "- (Button)",
              "Clear All (View)", "Brand (ImageView)", "Size And Type (ImageView)", "Release Year (ImageView)",
              "Product Types (ImageView)", "Price Range (ImageView)", "Color (ImageView)", "View Results (Button)",
              "Home Tab 1 of 5 (ImageView) selected", "Search Tab 2 of 5 (ImageView)",
              "Calendar Tab 3 of 5 (ImageView)", "Wishlist Tab 4 of 5 (ImageView)", "Account Tab 5 of 5 (ImageView)"]

state = {"goal": goal, "history": history, "ui": ["[%d] %s" % (i, c) for i, c in enumerate(candidates)]}
questions = {
    "operation": {"type": "choice", "instructions": Q_OP, "criteria": OP_DESC},
    "target": {"type": "choice", "instructions": Q_TARGET, "criteria": {str(i): c for i, c in enumerate(candidates)}},
}
out = agent.system_one(state, questions, max_len=3072, head_max_len=1536)
op, tgt = out["answers"]["operation"], out["answers"]["target"]
print(op["choice"], op["probabilities"][op["choice"]])                                   # CLICK 0.9347
print(tgt["choice"], candidates[int(tgt["choice"])], tgt["probabilities"][tgt["choice"]])  # 4 Brand (ImageView) 0.9313
