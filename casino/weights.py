"""Everything LoRA Casino knows about a checkpoint, worked out from the checkpoint's own weights (numpy only).

A "neuron" is one hidden unit of a transformer block's feed-forward layer (the input of its output weight). SDXL has
70 such layers, Anima 34 (families.py says which file is which and where the layers are). For the checkpoint in use
this module reads, once, and keeps in data/cache:

  absB   how big each neuron's activation typically is. Where a push is a bias change it moves a neuron by a multiple
         of this. Estimated from the weights: the layer's input comes out of a norm, so each pre-activation is roughly
         normal with a mean and spread given by that norm and the layer's input weights.
         SDXL: the unit is a * gelu(gate) of two of those. Checked against sizes measured on real pictures for one
         checkpoint: the per-layer level is within about +-40 %, single neurons are noisier (about x1.4 either way).
         Anima's text adapter: the unit is gelu of one. Not checked against measurements.
         Anima's blocks: 1, because a push there is a factor, not an amount.
  knn    each neuron's 12 nearest neighbours: the neurons of the same layer that write in the most similar direction
         (cosine of the columns of the layer's output weight).
  g1/g2  groups of about 1000 and pieces of about 100 neurons that write in similar directions (k-means on the same
         columns, per layer).

Nothing here is specific to one checkpoint: the layer list, sizes and all numbers come from the file that is read.
"""
import hashlib
import json
import os
import struct
import threading
from pathlib import Path

import numpy as np

from . import families

KNN = 12
VERSION = 1
CACHE = None  # set by app.configure(): <data>/cache
_hdr = {}
_lock = threading.Lock()
_state = {}   # cache key -> {"p": 0..1, "error": str | None, "stats": dict | None}


# ---------------------------------------------------------------- reading .safetensors
def header(path):
    """(data start, header) of a .safetensors file, cached."""
    st = os.stat(path)
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _hdr:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 200_000_000:
                raise ValueError(f"{Path(path).name} is not a .safetensors file")
            h = json.loads(f.read(n))
        if len(_hdr) > 8:
            _hdr.clear()
        _hdr[key] = (8 + n, h)
    return _hdr[key]


def tensor(path, name):
    """One tensor as float32 numpy, also for bf16 checkpoints."""
    base, h = header(path)
    t = h[name]
    a, b = t["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    dt = t["dtype"]
    if dt == "F16":
        arr = np.frombuffer(raw, np.float16)
    elif dt == "F32":
        arr = np.frombuffer(raw, np.float32)
    elif dt == "F64":
        arr = np.frombuffer(raw, np.float64)
    elif dt == "BF16":
        arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        raise ValueError(f"{name}: weights stored as {dt}, which LoRA Casino can't read")
    return arr.reshape(t["shape"]).astype(np.float32)


def model(path):
    """{"family", "prefix", "layers": [(name, neurons, width)]} of a model file; ValueError if it is not one we know."""
    _, h = header(path)
    return families.read({k: v for k, v in h.items() if k != "__metadata__"}, Path(path).name)


def layers(path):
    """[(name, neurons, width)] of the feed-forward layers, in network order."""
    return model(path)["layers"]


# ---------------------------------------------------------------- the numbers
def _erf(x):
    """erf, vectorised (Abramowitz & Stegun 7.1.26, error under 2e-7)."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


_GX, _GW = np.polynomial.hermite_e.hermegauss(48)
_GW = _GW / _GW.sum()


def _mean_abs_gelu(mu, sd):
    """Mean |gelu(z)| for z normal with these means and spreads."""
    g = mu[:, None] + sd[:, None] * _GX[None, :]
    return (np.abs(0.5 * g * (1 + _erf(g / np.sqrt(2)))) * _GW[None, :]).sum(1)


def _gelu_moments(mu, sd):
    """(mean, mean square) of gelu(z) for z normal with these means and spreads."""
    g = mu[:, None] + sd[:, None] * _GX[None, :]
    y = 0.5 * g * (1 + _erf(g / np.sqrt(2)))
    return (y * _GW[None, :]).sum(1), (y * y * _GW[None, :]).sum(1)


def _silu(x):
    return x / (1 + np.exp(-x))


# Anima draws a picture over timesteps from 1 (noise) down to 0; its blocks are read at a few of them and averaged
ANIMA_TIMES = (0.97, 0.9, 0.8, 0.65, 0.45, 0.25)


def anima_clock(path, prefix):
    """What Anima's blocks are told about the time, at each of ANIMA_TIMES: [(embedding, the part of the modulation all
    blocks share)], or None if the file has no such weights. As in ComfyUI's comfy/ldm/cosmos/predict2.py."""
    _, h = header(path)
    need = [prefix + k for k in ("t_embedder.1.linear_1.weight", "t_embedder.1.linear_2.weight", "t_embedding_norm.weight")]
    if any(k not in h for k in need):
        return None
    L1, L2, gn = (tensor(path, k).astype(np.float64) for k in need)
    half = gn.shape[0] // 2
    freq = np.exp(-np.log(10000) * np.arange(half) / half)
    out = []
    for t in ANIMA_TIMES:
        ts = np.concatenate([np.cos(t * freq), np.sin(t * freq)])
        out.append((ts / np.sqrt((ts ** 2).mean() + 1e-6) * gn, L2 @ _silu(L1 @ ts)))
    return out


def anima_block(path, prefix, name, W, clock):
    """How one Anima block's feed-forward runs, from the weights: (mean activation of each neuron, its variance, the
    gate on the block's output, the size of the block's whole output).

    A block has no bias, but it shifts and scales the input of its feed-forward and gates its output by the timestep
    (adaLN):  in = LayerNorm(x) * (1 + scale_t) + shift_t,  out = gate_t * (W @ gelu(W1 @ in)).  LayerNorm's output is
    about unit-normal, so neuron i's pre-activation is about normal(W1_i . shift_t, |W1_i * (1 + scale_t)|).
    The size of the whole output is its root mean square, counting the neurons as independent."""
    _, h = header(path)
    p = prefix + name
    W1 = tensor(path, p + ".mlp.layer1.weight").astype(np.float64)
    dim = W.shape[0]
    mods = []
    A, B = p + ".adaln_modulation_mlp.1.weight", p + ".adaln_modulation_mlp.2.weight"
    if clock and A in h and B in h:
        A, B = tensor(path, A).astype(np.float64), tensor(path, B).astype(np.float64)
        for emb, shared in clock:
            mod = B @ (A @ _silu(emb))
            mods.append(np.split(mod + shared if shared.shape == mod.shape else mod, 3))
    else:  # a file without the timestep weights: as if the block did not modulate
        mods.append((np.zeros(dim), np.zeros(dim), np.ones(dim)))
    e = np.zeros(W1.shape[0])
    e2, var, gate = e.copy(), e.copy(), np.zeros(dim)
    for shift, scale, g in mods:
        et, e2t = _gelu_moments(W1 @ shift, np.sqrt(((W1 * (1 + scale)[None, :]) ** 2).sum(1)) + 1e-9)
        e += et / len(mods)
        e2 += e2t / len(mods)
        var += np.maximum(e2t - et * et, 0) / len(mods)
        gate += g / len(mods)
    unit = float(np.sqrt(((np.linalg.norm(gate[:, None] * W, axis=0) ** 2) * e2).sum()))
    return e.astype(np.float32), var.astype(np.float32), gate.astype(np.float32), unit


def text_unit(path, prefix, name, W):
    """The size of the whole output of one block of Anima's text adapter (as anima_block's, without a gate)."""
    need = [prefix + name + k for k in (".mlp.0.weight", ".mlp.0.bias", ".norm_mlp.weight")]
    W0, b0, gam = (tensor(path, k).astype(np.float64) for k in need)
    _, e2 = _gelu_moments(b0, np.sqrt(((W0 * gam[None, :]) ** 2).sum(1)) + 1e-9)
    return float(np.sqrt(((np.linalg.norm(W, axis=0) ** 2) * e2).sum()))


def activation_size(path, family, prefix, name, inner):
    """Estimated mean |activation| of each neuron of one layer (see the module text); None if it can't be made."""
    _, h = header(path)
    p = prefix + name
    if family == "anima":
        if not name.startswith("llm_adapter."):
            return np.ones(inner, np.float32)       # a block's push is a factor
        need = [p + ".mlp.0.weight", p + ".mlp.0.bias", p + ".norm_mlp.weight"]
        if any(k not in h for k in need):
            return None
        W0, b0, gam = (tensor(path, k).astype(np.float64) for k in need)
        if W0.shape[0] != inner:
            return None
        sd = np.sqrt(((W0 * gam[None, :]) ** 2).sum(1)) + 1e-9   # the input is RMS-normed: each coordinate about gam
        return _mean_abs_gelu(b0, sd).astype(np.float32)
    need = [p + ".ff.net.0.proj.weight", p + ".ff.net.0.proj.bias", p + ".norm3.weight", p + ".norm3.bias"]
    if any(k not in h for k in need):
        return None
    W0, b0, gam, bet = (tensor(path, k).astype(np.float64) for k in need)
    if W0.shape[0] != 2 * inner:
        return None
    mu = W0 @ bet + b0
    sd = np.sqrt(((W0 * gam[None, :]) ** 2).sum(1)) + 1e-9
    mu_a, sd_a, mu_g, sd_g = mu[:inner], sd[:inner], mu[inner:], sd[inner:]
    z = mu_a / sd_a
    e_a = sd_a * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + mu_a * _erf(z / np.sqrt(2))  # mean |a| of a normal
    return (e_a * _mean_abs_gelu(mu_g, sd_g)).astype(np.float32)                        # x mean |gelu(gate)|


def _kmeans(X, k, rng, rounds=12):
    """Spherical k-means on unit rows X -> a cluster number per row."""
    if k <= 1:
        return np.zeros(len(X), np.int32)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    a = np.zeros(len(X), np.int32)
    for _ in range(rounds):
        a = (X @ C.T).argmax(1).astype(np.int32)
        for j in range(k):
            m = a == j
            c = X[m].sum(0) if m.any() else X[int(rng.integers(len(X)))]
            C[j] = c / (np.linalg.norm(c) + 1e-9)
    return a


def build(path, tick=lambda done, of: None):
    """Read the checkpoint and work everything out. tick(done, of) reports progress per layer."""
    M = model(path)
    L, fam, prefix = M["layers"], M["family"], M["prefix"]
    wkeys = families.per_layer(fam, prefix, [n for n, _, _ in L])["wkey"]
    inner = [n for _, n, _ in L]
    off = np.concatenate([[0], np.cumsum(inner)]).astype(np.int64)
    N = int(off[-1])
    absB = np.zeros(N, np.float32)
    knn = np.zeros((N, KNN), np.int32)
    g1 = np.zeros(N, np.int32)
    g2 = np.zeros(N, np.int32)
    n1 = n2 = 0
    # Anima only: a card's strength is counted against its block's whole output (see anima_block and engine.py)
    sized = fam == "anima"
    act, var = (np.zeros(N, np.float32), np.zeros(N, np.float32)) if sized else (None, None)
    gate = np.zeros((len(L), max(d for _, _, d in L)), np.float32) if sized else None
    unit = np.zeros(len(L), np.float32)
    clock = anima_clock(path, prefix) if sized else None
    for li, (name, n, _dim) in enumerate(L):
        a, b = int(off[li]), int(off[li + 1])
        W = tensor(path, wkeys[li])                                  # (width, neurons): column i = what neuron i writes
        X = np.ascontiguousarray((W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)).T)  # unit rows, one per neuron
        est = activation_size(path, fam, prefix, name, n)
        absB[a:b] = est if est is not None else 0.0
        if sized and name.startswith("blocks."):
            act[a:b], var[a:b], gate[li, :W.shape[0]], unit[li] = anima_block(path, prefix, name, W, clock)
        elif sized and est is not None:
            unit[li] = text_unit(path, prefix, name, W)
        k = min(KNN, n - 1)
        for r0 in range(0, n, 2048):                                 # in slices: a wide layer's full table is 250 MB
            S = X[r0:r0 + 2048] @ X.T
            S[np.arange(len(S)), np.arange(r0, r0 + len(S))] = -2.0  # not its own neighbour
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
            knn[a + r0:a + r0 + len(S), :k] = np.take_along_axis(top, order, 1) + a
            del S
        rng = np.random.default_rng(1000 + li)
        c1 = _kmeans(X, max(2, round(n / 1000)), rng)
        g1[a:b] = c1 + n1
        n1 += int(c1.max()) + 1
        c2 = _kmeans(X, max(2, round(n / 100)), rng)
        g2[a:b] = c2 + n2
        n2 += int(c2.max()) + 1
        tick(li + 1, len(L))
    ok = absB > 0
    if not ok.all():  # a layer the estimate could not be made for: use the typical size of the rest
        absB[~ok] = float(np.median(absB[ok])) if ok.any() else 0.05
    layer = np.repeat(np.arange(len(L)), inner).astype(np.int16)
    idx = (np.arange(N) - off[layer]).astype(np.int32)
    out = {"family": fam, "prefix": prefix, "names": [n for n, _, _ in L], "inner": inner, "off": off, "n": N,
           "layer": layer, "idx": idx, "absB": absB, "knn": knn, "g1": g1, "g2": g2, "unit": unit}
    if sized:
        out.update(act=act, var=var, gate=gate)
    return out


def _finish(st):
    """Add what follows from the family and the layer names: per layer, the weight to read, the LoRA key, the kind of
    push, the suit and the starting strengths (see families.py)."""
    st.update(families.per_layer(st["family"], st["prefix"], st["names"]))
    return st


def cache_file(path):
    s = os.stat(path)
    key = hashlib.sha1(f"{Path(path).resolve()}|{s.st_size}|{s.st_mtime_ns}".encode()).hexdigest()[:16]
    return key, Path(CACHE) / f"{key}_v{VERSION}.npz"


def _load(f):
    with np.load(f, allow_pickle=False) as z:  # closed before returning: an open file can't be replaced on Windows
        names = json.loads(str(z["names"]))
        inner = [int(x) for x in z["inner"]]
        fam = str(z["family"]) if "family" in z.files else "sdxl"      # files kept from before Anima are all SDXL
        prefix = str(z["prefix"]) if "prefix" in z.files else families.SDXL_PREFIX
        st = {"family": fam, "prefix": prefix,
              "names": names, "inner": inner, "off": z["off"].astype(np.int64), "n": int(z["off"][-1]),
              "layer": z["layer"], "idx": z["idx"], "absB": z["absB"], "knn": z["knn"], "g1": z["g1"], "g2": z["g2"],
              "unit": z["unit"] if "unit" in z.files else np.zeros(len(names), np.float32)}
        if fam == "anima":  # KeyError on a file kept before these were worked out: it is read again
            st.update(act=z["act"], var=z["var"], gate=z["gate"])
    return _finish(st)


def _work(path, key, f):
    s = _state[key]
    try:
        st = build(path, lambda done, of: s.update(p=round(done / of, 3)))
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, family=np.array(st["family"]), prefix=np.array(st["prefix"]),
                 names=np.array(json.dumps(st["names"])), inner=np.array(st["inner"]), off=st["off"], layer=st["layer"],
                 idx=st["idx"], absB=st["absB"], knn=st["knn"], g1=st["g1"], g2=st["g2"],
                 **{k: st[k] for k in ("unit", "act", "var", "gate") if k in st})
        os.replace(tmp, f)
        s.update(stats=_finish(st), p=1.0)
    except Exception as e:  # noqa: BLE001
        s.update(error=str(e) or type(e).__name__)


def state(path):
    """Where this checkpoint's numbers are: {"ready", "p", "error"}. Starts reading it if nobody has yet.
    Raises ValueError at once if the file is not a model we know."""
    model(path)
    key, f = cache_file(path)
    with _lock:
        s = _state.get(key)
        if s is not None and s["error"]:  # report a failed reading once; asking again tries again
            _state.pop(key)
            return {"ready": False, "p": s["p"], "error": s["error"]}
        if s is None:
            for k in [k for k, v in _state.items() if v["stats"] is not None][:-1]:  # keep two checkpoints in memory
                _state.pop(k)
            s = _state[key] = {"p": 0.0, "error": None, "stats": None}
            if f.exists():
                try:
                    s.update(stats=_load(f), p=1.0)
                except Exception:  # noqa: BLE001  a damaged or outdated cache file: read the checkpoint again
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass  # still open somewhere: the new reading replaces it when it is done
            if s["stats"] is None:
                threading.Thread(target=_work, args=(path, key, f), daemon=True, name="read-checkpoint").start()
    return {"ready": s["stats"] is not None, "p": s["p"], "error": s["error"]}


def stats(path):
    """The numbers if they are ready, else None (state() starts and reports the reading)."""
    key, _ = cache_file(path)
    s = _state.get(key)
    return s["stats"] if s else None
