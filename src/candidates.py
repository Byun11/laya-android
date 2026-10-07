"""UI candidates from an accessibility forest, and gold click -> candidate grounding.

Candidates never depend on the gold action: they are a pure function of (nodes, screen).
"""
from accessibility import IME_WINDOW, index, label, visible_box

ACTION_FLAGS = ("clickable", "long_clickable", "editable", "scrollable", "checkable")


def build(nodes, screen):
    """Deterministic candidate list: visible, actionable, non-IME nodes, deduplicated, in screen order.

    Dedup key = (bounds, label): a parent/child pair that renders as the same box with the same label
    collapses to the deepest node, with action flags OR-ed.
    """
    by_key = index(nodes)
    groups = {}
    for n in nodes:
        if n["window_type"] == IME_WINDOW or not any(n.get(f) for f in ACTION_FLAGS):
            continue
        box = visible_box(n, screen)
        if box is None:
            continue
        lab = label(n, by_key)
        key = (box, lab)
        c = groups.get(key)
        if c is None or n["depth"] > c["depth"]:
            merged = {f: n.get(f, False) or (c[f] if c else False) for f in ACTION_FLAGS}
            groups[key] = c = {
                "label": lab, "class": (n.get("class") or "").rsplit(".", 1)[-1],
                "resource_id": (n.get("resource_id") or "").rsplit("/", 1)[-1],
                "enabled": n.get("enabled", False), "checked": n.get("checked", False),
                "selected": n.get("selected", False), "bounds": list(box), "depth": n["depth"], **merged,
            }
        else:
            for f in ACTION_FLAGS:
                c[f] = c[f] or n.get(f, False)
    cands = sorted(groups.values(), key=lambda c: (c["bounds"][1], c["bounds"][0], c["bounds"][3],
                                                   c["bounds"][2], c["label"], c["class"]))
    for i, c in enumerate(cands):
        c["id"] = i
    return cands


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def ground(cands, x, y):
    """Index of the candidate a tap at (x, y) acts on, or None (ungroundable).

    Smallest containing box; ties -> deepest node; then lowest index (deterministic).
    """
    hits = [c for c in cands if c["bounds"][0] <= x <= c["bounds"][2] and c["bounds"][1] <= y <= c["bounds"][3]]
    if not hits:
        return None
    return min(hits, key=lambda c: (_area(c["bounds"]), -c["depth"], c["id"]))["id"]


if __name__ == "__main__":
    # self-check: dedup collapses parent/child with same box+label, grounding picks the smallest box
    nodes = [
        {"window": 0, "window_type": 1, "id": 1, "depth": 0, "visible": True, "clickable": True,
         "bounds": (0, 0, 100, 100), "children": [2]},
        {"window": 0, "window_type": 1, "id": 2, "depth": 1, "visible": True, "clickable": True,
         "bounds": (0, 0, 100, 100), "text": "", "children": [3]},
        {"window": 0, "window_type": 1, "id": 3, "depth": 2, "visible": True, "text": "OK",
         "bounds": (10, 10, 30, 30), "children": []},
        {"window": 0, "window_type": 1, "id": 4, "depth": 1, "visible": True, "clickable": True,
         "bounds": (40, 40, 60, 60), "content_desc": "Menu", "children": []},
        {"window": 0, "window_type": 2, "id": 5, "depth": 1, "visible": True, "clickable": True,
         "bounds": (0, 90, 10, 100), "text": "q", "children": []},
    ]
    cs = build(nodes, (100, 100))
    assert [c["label"] for c in cs] == ["OK", "Menu"], cs
    assert cs[0]["depth"] == 1
    assert ground(cs, 50, 50) == 1 and ground(cs, 5, 5) == 0
    assert ground([], 1, 1) is None
    print("ok")
