"""Accessibility-forest helpers: visibility and human-readable labels for nodes."""

IME_WINDOW = 2  # AndroidAccessibilityWindowInfo.WindowType.TYPE_INPUT_METHOD


def visible_box(node, screen):
    """Bounds clipped to the screen, or None when the node is invisible / zero-area / off-screen."""
    if not node.get("visible") or "bounds" not in node:
        return None
    w, h = screen
    x1, y1, x2, y2 = node["bounds"]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def own_text(node):
    return (node.get("text") or node.get("content_desc") or node.get("hint") or "").strip()


def label(node, by_key, limit=60):
    """Own text/description/hint, else the texts of its descendants (DFS order), capped at `limit` chars.

    A clickable row usually carries no text itself; its meaning lives in child TextViews.
    """
    t = own_text(node)
    if t:
        return " ".join(t.split())[:limit]
    parts, stack, seen = [], list(reversed(node.get("children", []))), set()
    while stack and sum(len(p) + 1 for p in parts) < limit:
        cid = stack.pop()
        child = by_key.get((node["window"], cid))
        if child is None or cid in seen:
            continue
        seen.add(cid)
        ct = own_text(child)
        if ct:
            parts.append(" ".join(ct.split()))
        stack.extend(reversed(child.get("children", [])))
    return " ".join(parts)[:limit]


def index(nodes):
    return {(n["window"], n.get("id")): n for n in nodes}
