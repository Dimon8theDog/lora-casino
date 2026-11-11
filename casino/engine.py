"""The game itself, with no ComfyUI in it (numpy only): hands of cards, the moves a deal makes, and turning a hand
into LoRA weights.

A hand (mix) = list of cards {ids: [global neuron ids], s: strength, label, kind, center, held?}. Each neuron of a
card is moved by s x its typical activation size (weights.py works that out from the checkpoint). Why a LoRA can hold
a hand exactly: a push adds a fixed vector d to the input of a feed-forward output layer (ff.net.2: out = W x + b),
and W (x + d) + b = W x + (b + W d), so the push is the same as changing that layer's bias by W d. That is saved as
standard 'diff_b' LoRA keys, which ComfyUI's LoRA loaders apply. Strength 1 = the picture; it is linear.

W is the checkpoint's own, so a saved LoRA is exact on the checkpoint it was made with and close on its relatives.
"""
import hashlib
import json
import os
import struct
import uuid
import zlib

import numpy as np

from . import weights

# SDXL's own structure (the same in every SDXL checkpoint): the middle of the decoder sets the look, its last,
# high-resolution blocks paint fine detail, everything before lays the picture out
STYLE_BLOCKS = ("output_blocks.0.1", "output_blocks.1.1", "output_blocks.2.1")


DETAIL_BLOCKS = ("output_blocks.3.1", "output_blocks.4.1", "output_blocks.5.1")


DEFAULT_S = {1: 300.0, 10: 100.0, 100: 60.0, 1000: 6.0}   # push strengths (x typical activation) that show in a picture


def where(St, li):
    b = St["blocks"][li]
    if b in STYLE_BLOCKS:
        return "style"
    return "fine detail" if b.startswith("output_blocks") else "layout"


def label_ids(St, g):
    return f"L{int(St['layer'][g])}:{int(St['idx'][g])}"


def size_of(m):
    return 1000 if m["kind"] == "G" else 100 if m["kind"] == "S" or len(m["ids"]) > 10 else 10 if len(m["ids"]) > 1 else 1


def pick(St, size, seed_g=None, rng=None, sign=None):
    """A push of about `size` neurons around a random (or given) neuron; random sign unless given."""
    rng = rng or np.random.default_rng()
    g = int(seed_g) if seed_g is not None else int(rng.integers(St["n"]))
    sign = sign if sign in (1, -1) else (1 if rng.random() < 0.5 else -1)
    if size >= 1000:
        G = int(St["g1"][g])
        ids, label, kind = np.where(St["g1"] == G)[0], f"G{G}", "G"
    elif size >= 100:
        S = int(St["g2"][g])
        ids, label, kind = np.where(St["g2"] == S)[0], f"S{S}", "S"
    elif size >= 10:
        ids = np.concatenate([[g], St["knn"][g][:9]])
        label, kind = f"{label_ids(St, g)} +9 near", "cluster"
    else:
        ids, label, kind = np.array([g]), label_ids(St, g), "neuron"
    return {"ids": [int(i) for i in ids], "label": label, "kind": kind, "center": g,
            "s": sign * DEFAULT_S[1000 if size >= 1000 else 100 if size >= 100 else 10 if size >= 10 else 1]}


def near(St, g, rng=None):
    """A neuron close to g (a neighbour, or a neighbour's neighbour)."""
    rng = rng or np.random.default_rng()
    n = int(rng.choice(St["knn"][g]))
    return int(rng.choice(St["knn"][n])) if rng.random() < 0.5 else n


def clean_mix(mix, N=None):
    N = N or 10 ** 9
    out = []
    for m in mix or []:
        if not isinstance(m, dict):
            continue
        ids = [int(i) for i in m.get("ids", []) if 0 <= int(i) < N]
        s = float(m.get("s", 0))
        if ids and s and np.isfinite(s) and abs(s) <= 5000:
            c = m.get("center")
            c = int(c) if isinstance(c, int) and 0 <= c < N else ids[0]
            out.append({"ids": ids, "s": s, "label": str(m.get("label", ""))[:60], "kind": str(m.get("kind", ""))[:12], "center": c}
                       | ({"held": True} if m.get("held") is True else {}))  # a held card: deals never change it
    if sum(len(m["ids"]) for m in out) > 40000:
        raise ValueError("mix too big (over 40,000 neurons)")
    return out


def shifts(St, mix):
    """{layer: dense shift vector} for a mix."""
    out = {}
    for m in mix:
        ids = np.array(m["ids"])
        for li in np.unique(St["layer"][ids]):
            sel = ids[St["layer"][ids] == li]
            v = out.setdefault(int(li), np.zeros(St["inner"][int(li)], np.float32))
            np.add.at(v, St["idx"][sel].astype(np.int64), m["s"] * St["absB"][sel])
    return out


_W = {}


def weight(St, li, ckpt):
    """FF output weight of layer li from the checkpoint (a few layers kept in memory)."""
    key = (str(ckpt), li)
    if key not in _W:
        if len(_W) > 16:
            _W.pop(next(iter(_W)))
        base = St["names"][li] + ".ff"
        _W[key] = (base, weights.tensor(ckpt, f"{weights.PREFIX}{base}.net.2.weight"))
    return _W[key]


def lora_tensors(St, mix, ckpt):
    out, info = {}, []
    for li, v in sorted(shifts(St, mix).items()):
        base, W = weight(St, li, ckpt)
        db = W @ v
        out[f"lora_unet_{(base + '.net.2').replace('.', '_')}.diff_b"] = db.astype(np.float32)
        info.append({"layer": li, "name": base, "neurons": int((v != 0).sum()), "bias_change_norm": round(float(np.linalg.norm(db)), 3)})
    return out, info


def recipe_text(St, mix):
    """The mix written out: neurons per layer and how far each push moves them."""
    parts = []
    for m in mix:
        ids = np.array(m["ids"])
        for li in np.unique(St["layer"][ids]):
            sel = ids[St["layer"][ids] == li]
            parts.append(f"L{int(li)}:{';'.join(str(int(i)) for i in St['idx'][sel])} {m['s']:g}")
    return ", ".join(parts)


def mix_key(St, mix):
    h = hashlib.sha1()
    for li, v in sorted(shifts(St, mix).items()):
        nz = np.nonzero(v)[0]
        h.update(f"{li}:".encode() + nz.astype(np.int32).tobytes() + np.round(v[nz], 4).astype(np.float32).tobytes())
    return h.hexdigest()[:12]


def save_safetensors(path, tensors, metadata):
    """Write float32 tensors as a .safetensors file (the format is a JSON header and the raw numbers)."""
    head, blobs, at = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}, [], 0
    for k in sorted(tensors):
        b = np.ascontiguousarray(tensors[k], dtype="<f4").tobytes()
        head[k] = {"dtype": "F32", "shape": list(tensors[k].shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    hj = json.dumps(head, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def write_lora(path, St, mix, ckpt, meta=None):
    tensors, info = lora_tensors(St, mix, ckpt)
    md = {"ss_network_module": "loracasino.bias_diff", "ss_base_model_version": "sdxl_base_v1-0",
          "modelspec.architecture": "stable-diffusion-xl-v1-base/lora"}
    md.update(meta or {})
    save_safetensors(path, tensors, md)
    return info


def png_with_text(png, texts):
    """The same PNG with text chunks added after its header (ComfyUI reads 'prompt' from there on drag-in)."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        return png
    end = 8 + 12 + struct.unpack(">I", png[8:12])[0]  # after IHDR
    extra = b""
    for k, v in texts.items():
        body = k.encode("latin-1") + b"\x00" + v.encode("latin-1", "replace")
        extra += struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    return png[:end] + extra + png[end:]


SIZES = (1, 10, 100, 1000)


# the five sliders (0..1) that steer which pushes get picked
KNOBS = {"size": 0.7, "strength": 0.5, "depth": 0.72, "spread": 0.25, "count": 0.35}


_pw = {}


def describe(St, m):
    """One push in plain words."""
    d0 = DEFAULT_S[size_of(m)]
    r = abs(m["s"]) / d0
    power = "gentle" if r < 0.35 else "medium" if r < 0.65 else "strong" if r < 1.0 else "very strong"
    c = int(m.get("center", m["ids"][0]))
    what = {"G": f"group {m['label']} ({len(m['ids'])} neurons)", "S": f"piece {m['label']} ({len(m['ids'])} neurons)",
            "cluster": f"{len(m['ids'])} neurons near {m['label'].split()[0]}", "neuron": f"neuron {m['label']}"}.get(m["kind"], m["label"])
    cid = hashlib.sha1(np.sort(np.array(m["ids"], np.int32)).tobytes()).hexdigest()[:8]  # the card: which neurons, not how hard
    return {"cid": cid, "what": what, "up": m["s"] > 0, "s": round(float(m["s"]), 3), "power": power, "n": len(m["ids"]),
            "where": where(St, int(St["layer"][c])), "held": bool(m.get("held"))}


def suit(St, m):
    """The kind of card a push is, by where its neurons sit: layout, style or detail."""
    w = where(St, int(St["layer"][int(m.get("center", m["ids"][0]))]))
    return "detail" if w == "fine detail" else w


def knobs(body):
    k = dict(KNOBS)
    for n, v in (body.get("knobs") or {}).items():
        if n in k:
            try:
                k[n] = min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                pass
    return k


def pick_weights(St, kn):
    """Cumulative pick chances over all neurons for the depth / spread sliders.
    depth: 0 = first (layout) layers .. ~0.7 = the style layers .. 1 = the last (fine detail) layers;
    spread: how far from that depth picks may land (1 = anywhere)."""
    key = (id(St), round(kn["depth"], 2), round(kn["spread"], 2))
    if key not in _pw:
        L = len(St["names"])
        pos = np.arange(L) / (L - 1)
        sd = 0.04 + 0.6 * kn["spread"]
        lw = np.exp(-((pos - kn["depth"]) ** 2) / (2 * sd * sd))
        w = lw[St["layer"]].astype(np.float64)
        if len(_pw) > 64:
            _pw.clear()
        _pw[key] = np.cumsum(w / w.sum())
    return _pw[key]


def detail_weights(St):
    key = (id(St), "detail")
    if key not in _pw:
        on = np.array([b in DETAIL_BLOCKS for b in St["blocks"]])
        w = on[St["layer"]].astype(np.float64)
        _pw[key] = np.cumsum(w / w.sum()) if w.sum() else np.cumsum(np.full(St["n"], 1.0 / St["n"]))
    return _pw[key]


def draw_from(cum, rng, N):
    return min(N - 1, int(np.searchsorted(cum, rng.random())))


def ingredient(St, rng, wild, kn):
    """A fresh random push, steered by the sliders: size (one neuron .. a whole ~1000 group), strength, where in the
    network. Any sign. wild nudges toward bigger and stronger."""
    c = kn["size"] * 3 + 0.4 * (wild - 0.5)
    p_size = np.exp(-((np.arange(4) - c) ** 2) / (2 * 0.7 ** 2))
    size = int(rng.choice(SIZES, p=p_size / p_size.sum()))
    p = pick(St, size, draw_from(pick_weights(St, kn), rng, St["n"]), rng)
    p["s"] *= (float(np.exp(rng.uniform(np.log(0.55), np.log(1.6)))) * strength_of(kn)
               * (0.75 + 0.5 * wild) * (0.8 if size == 1000 else 1.0))
    return p


def strength_of(kn):
    """The 'push strength' slider as a multiple of DEFAULT_S: 0 = x0.2, middle = x0.5, 1 = x1.2.
    The middle is half of what a sturdy checkpoint takes, because checkpoints differ a lot in how hard they can be
    pushed before the picture breaks (seen: one Illustrious merge fine at x1, the Illustrious base broken at x0.5)."""
    return 0.5 * 6.0 ** (kn["strength"] - 0.5)


def n_fresh(rng, kn, extra=0):
    """How many pushes a fresh look gets ('pushes per new look' slider: 0 = 1, 1 = about 5)."""
    return int(np.clip(round(1 + kn["count"] * 4 + rng.uniform(-1, 1)) + extra, 1, 7))


def move_weights(wild, n_parent, n_free=None):
    """How likely each kind of move is. Low stakes = small edits of the hand; high = remixes and new hands.
    n_free = the cards that are not held: only those can be changed."""
    if n_parent == 0:
        return {"leap": 1.0}
    if n_free == 0:
        return {"add": 1.0, "leap": 0.2 + 0.6 * wild}
    w = {"tweak": 0.15 + 1.5 * (1 - wild), "flip": 1.0, "add": 0.9 + 0.5 * wild, "swap": 1.0,
         "drop": 0.7 if n_parent > 1 else 0.0, "remix": 0.15 + 1.6 * wild, "leap": 0.03 + 2.2 * wild ** 2}
    if n_parent >= 7:
        w["add"] *= 0.25
    return w


def deck(n, w, rng):
    """n moves for one round, never more than a third (min 2) of one kind, so a round always mixes kinds."""
    cap = max(2, -(-n // 3)) if len([k for k, v in w.items() if v > 0]) > 1 else n
    out = []
    for _ in range(n):
        ks = [k for k, v in w.items() if v > 0 and out.count(k) < cap] or [k for k, v in w.items() if v > 0]
        p = np.array([w[k] for k in ks])
        out.append(ks[int(rng.choice(len(ks), p=p / p.sum()))])
    return out


def apply_move(St, parent, move, rng, wild, kn):
    """-> (child mix, caption)."""
    child = [dict(m) for m in parent]
    held = [m for m in child if m.get("held")]
    free = [i for i, m in enumerate(child) if not m.get("held")]
    if move == "leap":
        new = [ingredient(St, rng, wild, kn) for _ in range(n_fresh(rng, kn, -len(held)))]
        return held + new, f"kept the held, {len(new)} new" if held else "a new hand"
    if move == "add" or not free:
        p = ingredient(St, rng, wild, kn)
        child.append(p)
        return child, f"drew a {suit(St, p)} card"
    if move == "remix":
        keep = [child[i] for i in free if rng.random() < 0.5]
        if not keep and not held:
            keep = [child[free[int(rng.integers(len(free)))]]]
        new = [ingredient(St, rng, wild, kn) for _ in range(max(1, n_fresh(rng, kn, -1) // 2 + int(rng.random() < 0.4 * wild)))]
        return held + keep + new, f"kept {len(held) + len(keep)}, drew {len(new)}"
    k = free[int(rng.integers(len(free)))]
    m = child[k]
    if move == "drop":
        if len(child) < 2:
            return child, ""
        child.pop(k)
        return child, f"discarded a {suit(St, m)} card"
    if move == "flip":
        m["s"] = -m["s"]
        return child, f"turned a {suit(St, m)} card over"
    if move == "tweak":
        f = float(np.exp(rng.uniform(np.log(1.25), np.log(1.7 + 0.9 * wild))))
        if rng.random() < 0.5:
            m["s"] *= f
            return child, f"raised a {suit(St, m)} card"
        m["s"] /= f
        return child, f"lowered a {suit(St, m)} card"
    # swap: the same kind of card, around a neighbour of it
    p = pick(St, size_of(m), near(St, int(m.get("center", m["ids"][0])), rng), rng, sign=1 if m["s"] > 0 else -1)
    p["s"] = m["s"]
    child[k] = p
    return child, f"swapped a {suit(St, m)} card"


def mutate(St, parent, move, rng, wild, kn):
    """One main move, plus (more often when wild) a small extra edit."""
    child, cap = apply_move(St, parent, move, rng, wild, kn)
    caps = [cap]
    if move not in ("leap",) and child and rng.random() < 0.15 + 0.45 * wild:
        extra = [x for x in ("tweak", "add", "swap") if x != move]
        child, cap2 = apply_move(St, child, str(rng.choice(extra)), rng, wild, kn)
        caps.append(cap2)
    return child, [c for c in caps if c]


def twin(St, m):
    """The closest push of the same kind: the nearest neighbour of its centre that lands in another group/piece."""
    c = int(m.get("center", m["ids"][0]))
    for n in St["knn"][c]:
        p = pick(St, size_of(m), int(n), sign=1 if m["s"] > 0 else -1)
        if p["label"] != m["label"] and set(p["ids"]) != set(m["ids"]):
            p["s"] = m["s"]
            return p
    return None


def polish(St, parent, n, rng, fine):
    """Small fixing moves around a look you like. fine 0..1: 0 = tiniest (about 6%), 1 = bigger (about 30%).
    Kinds: one push weaker / stronger, the look without one push, one push -> its closest twin, everything a bit
    weaker / stronger, a tiny touch in the last (fine detail) layers. Mixed so a step covers several kinds and pushes."""
    step = 0.06 + 0.24 * fine
    kinds = {"weaker": [], "stronger": [], "without": [], "twin": [], "all": [], "touch": []}
    for k, m in enumerate(parent):
        if m.get("held"):  # a held card stays exactly as it is
            continue
        nm = f"a {suit(St, m)} card"
        f = 1 + step * rng.uniform(0.75, 1.3)
        pct = round((f - 1) * 100)
        c = [dict(x) for x in parent]
        c[k]["s"] /= f
        kinds["weaker"].append((c, f"lowered {nm} {pct}%"))
        c = [dict(x) for x in parent]
        c[k]["s"] *= f
        kinds["stronger"].append((c, f"raised {nm} {pct}%"))
        if len(parent) > 1:
            kinds["without"].append(([dict(x) for j, x in enumerate(parent) if j != k], f"without {nm}"))
        t = twin(St, m) if (m["kind"] != "G" or fine >= 0.5) else None  # swapping a whole group is not a small fix
        if t:
            c = [dict(x) for x in parent]
            c[k] = t
            kinds["twin"].append((c, f"{nm} for its twin"))
    if any(not x.get("held") for x in parent):
        for sgn in (-1, 1):
            f = 1 + step * 0.8 * rng.uniform(0.8, 1.2)
            kinds["all"].append(([x if x.get("held") else dict(x, s=x["s"] * (f if sgn > 0 else 1 / f)) for x in parent],
                                 f"{'raised' if sgn > 0 else 'lowered'} every card {round((f - 1) * 100)}%"))
    for _ in range(3):
        p = pick(St, 1 if rng.random() < 0.6 else 10, draw_from(detail_weights(St), rng, St["n"]), rng)
        p["s"] *= 0.2 + 0.4 * fine
        kinds["touch"].append(([dict(x) for x in parent] + [p], "drew a small detail card"))
    for v in kinds.values():
        rng.shuffle(v)
    order = [k for k in ("without", "weaker", "stronger", "twin", "all", "touch") if kinds[k]]
    rng.shuffle(order)
    out = []
    while len(out) < n and any(kinds[k] for k in order):  # one of each kind in turn
        for k in order:
            if kinds[k] and len(out) < n:
                mix, cap = kinds[k].pop()
                out.append((mix, [cap], "polish-" + k))
    return out
