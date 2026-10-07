"""AndroidControl reader: GZIP TFRecord shards -> episode dicts, with no TensorFlow / android_env.

ponytail: hand-rolled protobuf wire decoder; the global env's TF 2.15 can't import (numpy 2 / protobuf 6).
Field numbers: android_env/proto/a11y/*.proto (research/02-data-eval.md §1.3).
"""
import json
import os
import struct
import zlib

DATA_ROOT = os.environ.get("LAYA_ANDROID_DATA", "D:/laya-android")
RAW_DIR = os.path.join(DATA_ROOT, "raw", "android_control")


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = _varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError("wire type %d" % w)
        yield f, w, v


def records(path):
    """Raw TFRecord payloads from a GZIP shard, streamed."""
    d = zlib.decompressobj(31)
    buf = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            buf += d.decompress(chunk)
            while len(buf) >= 12:
                (ln,) = struct.unpack("<Q", buf[:8])
                if len(buf) < 16 + ln:
                    break
                yield buf[12:12 + ln]
                buf = buf[16 + ln:]
    if buf:
        raise ValueError("%s: %d trailing bytes (truncated shard?)" % (path, len(buf)))


def _example(rec, skip=("screenshots",)):
    """tf.train.Example -> {key: [bytes | int]}; screenshots skipped (v0 is text-only)."""
    out = {}
    for _, _, feats in _fields(rec):
        for _, _, entry in _fields(feats):
            key = val = None
            for ef, _, ev in _fields(entry):
                if ef == 1:
                    key = ev.decode()
                else:
                    val = ev
            if key in skip:
                continue
            for lf, _, lv in _fields(val):
                if lf == 1:  # bytes_list
                    out[key] = [v for _, _, v in _fields(lv)]
                elif lf == 3:  # int64_list (packed or not)
                    vals = []
                    for _, w, v in _fields(lv):
                        if w == 2:
                            j = 0
                            while j < len(v):
                                x, j = _varint(v, j)
                                vals.append(x)
                        else:
                            vals.append(v)
                    out[key] = vals
    return out


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


_STR = {3: "class", 4: "content_desc", 5: "hint", 6: "package", 7: "text", 10: "resource_id"}
_BOOL = {12: "checkable", 13: "checked", 14: "clickable", 15: "editable", 16: "enabled", 17: "focusable",
         18: "focused", 19: "long_clickable", 20: "password", 21: "scrollable", 22: "selected", 23: "visible"}


def parse_forest(blob):
    """AndroidAccessibilityForest -> flat node list. Each node: id, window, window_type, bounds, depth,
    children, string fields from _STR, bool flags from _BOOL (absent = False)."""
    nodes = []
    for widx, (_, _, win) in enumerate(_fields(blob)):  # windows = 1
        wtype = 0
        trees = []
        for wf, _, wv in _fields(win):
            if wf == 6:
                wtype = wv
            elif wf == 11:
                trees.append(wv)
        for tree in trees:
            for _, _, nb in _fields(tree):  # nodes = 1
                n = {"window": widx, "window_type": wtype, "children": [], "depth": 0}
                for nf, w, nv in _fields(nb):
                    if nf == 1:
                        n["id"] = nv
                    elif nf == 2:
                        r = {1: 0, 2: 0, 3: 0, 4: 0}
                        for rf, _, rv in _fields(nv):
                            r[rf] = _signed(rv)
                        n["bounds"] = (r[1], r[2], r[3], r[4])
                    elif nf in _STR:
                        n[_STR[nf]] = nv.decode("utf8", "replace")
                    elif nf in _BOOL:
                        n[_BOOL[nf]] = bool(nv)
                    elif nf == 25:  # child_ids: unpacked varints in this data, packed allowed
                        if w == 2:
                            j = 0
                            while j < len(nv):
                                x, j = _varint(nv, j)
                                n["children"].append(x)
                        else:
                            n["children"].append(nv)
                    elif nf == 27:
                        n["depth"] = nv
                nodes.append(n)
    return nodes


def episodes(path):
    """Episode dicts: episode_id, goal, step_instructions, actions (parsed JSON), trees (raw bytes, T+1)."""
    for rec in records(path):
        e = _example(rec)
        yield {
            "episode_id": e["episode_id"][0],
            "goal": e["goal"][0].decode("utf8"),
            "step_instructions": [s.decode("utf8") for s in e.get("step_instructions", [])],
            "actions": [json.loads(a) for a in e.get("actions", [])],
            "trees": e["accessibility_trees"],
            "screen": (e["screenshot_widths"][0], e["screenshot_heights"][0]),
        }


OPS = ["CLICK", "LONG_PRESS", "SCROLL_UP", "SCROLL_DOWN", "SCROLL_LEFT", "SCROLL_RIGHT",
       "OPEN_APP", "INPUT_TEXT", "BACK", "HOME", "WAIT", "DONE"]
_SIMPLE = {"click": "CLICK", "long_press": "LONG_PRESS", "open_app": "OPEN_APP", "input_text": "INPUT_TEXT",
           "navigate_back": "BACK", "navigate_home": "HOME", "wait": "WAIT"}


def canonical_op(action):
    """AndroidControl action JSON -> canonical op (scroll direction folded into the op)."""
    t = action["action_type"]
    if t == "scroll":
        return "SCROLL_" + action["direction"].upper()
    if t not in _SIMPLE:
        raise ValueError("unknown action_type %r" % t)
    return _SIMPLE[t]


def shard_paths():
    return [os.path.join(RAW_DIR, "android_control-%05d-of-00020" % i) for i in range(20)]


def load_splits():
    """{episode_id: split} with split in train/validation/test, plus {subsplit: set(ids)} for test."""
    with open(os.path.join(RAW_DIR, "splits.json")) as f:
        sp = json.load(f)
    with open(os.path.join(RAW_DIR, "test_subsplits.json")) as f:
        sub = json.load(f)
    split_of = {}
    for name, ids in sp.items():
        for i in ids:
            assert i not in split_of, "episode %s in two splits" % i
            split_of[i] = name
    return split_of, {k: set(v) for k, v in sub.items()}
