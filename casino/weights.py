"""Everything LoRA Casino knows about a checkpoint, worked out from the checkpoint's own weights (numpy only).

A "neuron" is one hidden unit of a transformer block's feed-forward layer (the input of ff.net.2). SDXL has 70 such
layers. For the checkpoint in use this module reads, once, and keeps in data/cache:

  absB   how big each neuron's activation typically is. A push moves a neuron by a multiple of this. Estimated from
         the weights: the layer's input comes out of a LayerNorm (norm3), so each pre-activation is roughly normal
         with a mean and spread given by norm3 and ff.net.0.proj; the unit is a * gelu(gate) of two of those.
         Checked against sizes measured on real pictures for one checkpoint: the per-layer level is within about
         +-40 %, single neurons inside a layer are noisier (about x1.4 either way).
  knn    each neuron's 12 nearest neighbours: the neurons of the same layer that write in the most similar direction
         (cosine of their ff.net.2 columns).
  g1/g2  groups of about 1000 and pieces of about 100 neurons that write in similar directions (k-means on the same
         columns, per layer).

Nothing here is specific to one checkpoint: the layer list, sizes and all numbers come from the file that is read.
"""
import hashlib
import json
import os
import re
import struct
import threading
from pathlib import Path

import numpy as np

PREFIX = "model.diffusion_model."
FF_OUT = re.compile(r"^model\.diffusion_model\.((input_blocks|middle_block|output_blocks)\.(?:(\d+)\.)?(\d+)"
                    r"\.transformer_blocks\.(\d+))\.ff\.net\.2\.weight$")
SDXL_LAYERS = 70
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


def layers(path):
    """[(name, neurons, width)] of the feed-forward layers, in network order; ValueError if this is not SDXL."""
    _, h = header(path)
    found = []
    for k, t in h.items():
        m = FF_OUT.match(k)
        if m:
            part = {"input_blocks": 0, "middle_block": 1, "output_blocks": 2}[m.group(2)]
            found.append(((part, int(m.group(3) or 0), int(m.group(4)), int(m.group(5))), m.group(1), t["shape"][1], t["shape"][0]))
    found.sort()
    name = Path(path).name
    if not found:
        raise ValueError(f"{name} has no SDXL UNet inside (a checkpoint for another kind of model, or a LoRA / VAE file)")
    if len(found) != SDXL_LAYERS:
        raise ValueError(f"{name} is not an SDXL checkpoint: it has {len(found)} transformer layers, SDXL has {SDXL_LAYERS}")
    return [(n, int(inner), int(dim)) for _, n, inner, dim in found]


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


def activation_size(path, name, inner):
    """Estimated mean |activation| of each neuron of one layer (see the module text)."""
    _, h = header(path)
    p = PREFIX + name
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
    g = mu_g[:, None] + sd_g[:, None] * _GX[None, :]
    e_g = (np.abs(0.5 * g * (1 + _erf(g / np.sqrt(2)))) * _GW[None, :]).sum(1)          # mean |gelu(gate)|
    return (e_a * e_g).astype(np.float32)


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
    L = layers(path)
    inner = [n for _, n, _ in L]
    off = np.concatenate([[0], np.cumsum(inner)]).astype(np.int64)
    N = int(off[-1])
    absB = np.zeros(N, np.float32)
    knn = np.zeros((N, KNN), np.int32)
    g1 = np.zeros(N, np.int32)
    g2 = np.zeros(N, np.int32)
    n1 = n2 = 0
    for li, (name, n, _dim) in enumerate(L):
        a, b = int(off[li]), int(off[li + 1])
        W = tensor(path, f"{PREFIX}{name}.ff.net.2.weight")          # (width, neurons): column i = what neuron i writes
        X = np.ascontiguousarray((W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)).T)  # unit rows, one per neuron
        est = activation_size(path, name, n)
        absB[a:b] = est if est is not None else 0.0
        S = X @ X.T
        np.fill_diagonal(S, -2.0)
        k = min(KNN, n - 1)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        knn[a:b, :k] = np.take_along_axis(top, order, 1) + a
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
    return {"names": [n for n, _, _ in L], "inner": inner, "off": off, "n": N, "layer": layer, "idx": idx,
            "absB": absB, "knn": knn, "g1": g1, "g2": g2}


def _finish(st):
    st["blocks"] = [n.split(".transformer")[0] for n in st["names"]]
    return st


def cache_file(path):
    s = os.stat(path)
    key = hashlib.sha1(f"{Path(path).resolve()}|{s.st_size}|{s.st_mtime_ns}".encode()).hexdigest()[:16]
    return key, Path(CACHE) / f"{key}_v{VERSION}.npz"


def _load(f):
    z = np.load(f, allow_pickle=False)
    names = json.loads(str(z["names"]))
    inner = [int(x) for x in z["inner"]]
    return _finish({"names": names, "inner": inner, "off": z["off"].astype(np.int64), "n": int(z["off"][-1]),
                    "layer": z["layer"], "idx": z["idx"], "absB": z["absB"], "knn": z["knn"], "g1": z["g1"], "g2": z["g2"]})


def _work(path, key, f):
    s = _state[key]
    try:
        st = build(path, lambda done, of: s.update(p=round(done / of, 3)))
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, names=np.array(json.dumps(st["names"])), inner=np.array(st["inner"]), off=st["off"], layer=st["layer"],
                 idx=st["idx"], absB=st["absB"], knn=st["knn"], g1=st["g1"], g2=st["g2"])
        os.replace(tmp, f)
        s.update(stats=_finish(st), p=1.0)
    except Exception as e:  # noqa: BLE001
        s.update(error=str(e) or type(e).__name__)


def state(path):
    """Where this checkpoint's numbers are: {"ready", "p", "error"}. Starts reading it if nobody has yet.
    Raises ValueError at once if the file is not an SDXL checkpoint."""
    layers(path)
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
                except Exception:  # noqa: BLE001  a damaged cache file: read the checkpoint again
                    f.unlink(missing_ok=True)
            if s["stats"] is None:
                threading.Thread(target=_work, args=(path, key, f), daemon=True, name="read-checkpoint").start()
    return {"ready": s["stats"] is not None, "p": s["p"], "error": s["error"]}


def stats(path):
    """The numbers if they are ready, else None (state() starts and reports the reading)."""
    key, _ = cache_file(path)
    s = _state.get(key)
    return s["stats"] if s else None
