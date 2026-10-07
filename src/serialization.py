"""Processed step item -> Laya `{state, questions, expected}` row.

Main setting: high-level goal only. `low_level=True` adds the step instruction (oracle diagnostic only).
"""
OP_DESC = {
    "CLICK": "tap a UI element",
    "LONG_PRESS": "long-press a UI element",
    "SCROLL_UP": "scroll up",
    "SCROLL_DOWN": "scroll down",
    "SCROLL_LEFT": "scroll left",
    "SCROLL_RIGHT": "scroll right",
    "OPEN_APP": "launch an app by name",
    "INPUT_TEXT": "type text into the focused field",
    "BACK": "press the system back button",
    "HOME": "go to the home screen",
    "WAIT": "wait for the screen to update",
    "DONE": "the goal is complete; stop",
}
Q_OP = "Which operation should be performed next?"
Q_TARGET = "Which UI element should be targeted?"


def option_label(c):
    flags = [f for f, on in (("edit", c["editable"]), ("long", c["long_clickable"] and not c["clickable"]),
                             ("scroll", c["scrollable"] and not c["clickable"]), ("on", c["checked"]),
                             ("selected", c["selected"]), ("disabled", not c["enabled"])) if on]
    return " ".join([c["label"][:40] or "-", "(%s)" % c["class"], *flags])


def row(item, low_level=False):
    state = {"goal": item["goal"]}
    if low_level:
        state["step_instruction"] = item["step_instruction"]
    labels = [option_label(c) for c in item["candidates"]]
    state["history"] = item["history"]
    state["ui"] = ["[%d] %s" % (i, lab) for i, lab in enumerate(labels)]
    questions = {"operation": {"type": "choice", "instructions": Q_OP, "criteria": dict(OP_DESC)}}
    expected = {"operation": item["operation_gold"]}
    if item["target_groundable"]:
        questions["target"] = {"type": "choice", "instructions": Q_TARGET,
                               "criteria": {str(i): lab for i, lab in enumerate(labels)}}
        expected["target"] = str(item["target_gold"])
    return {"state": state, "questions": questions, "expected": expected}
