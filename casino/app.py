"""LoRA Casino's requests: dealing a round, drawing it through ComfyUI, cashing a hand out as a LoRA.

The same code serves both ways of running it: inside ComfyUI (the plugin mounts handle() and post() on ComfyUI's own
server) and on its own (standalone.py serves them). configure() says where the page and the data folder are.

GET  /  /api/meta  /api/models  /api/model?ckpt=  /api/round?id=  /api/job?id=  /api/preview?pid=  /api/render/<key>.png
POST /api/breed  /api/breed/stop  /api/generate  /api/save  /api/settings
"""
import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import uuid
from pathlib import Path

import numpy as np

from . import host, weights
from .engine import (clean_mix, deck, describe, knobs, mix_key, move_weights, mutate, png_with_text, polish, recipe_text,
                     write_lora, KNOBS)
from .host import (CID, PREVIEW, ComfyDown, _live, ckpt_path, comfy, comfy_try, find_comfy, list_models, live, live_start,
                   lora_dir, set_settings, settings)

PAGE = RENDERS = SAVED = None


def configure(data_dir, page_dir, own_url=None, folders=None):
    """Where things are. own_url / folders are given when running inside ComfyUI (see host.py)."""
    global PAGE, RENDERS, SAVED
    data = Path(data_dir)
    PAGE, RENDERS, SAVED = Path(page_dir), data / "renders", data / "saved"
    weights.CACHE = data / "cache"
    host.DATA, host.OWN_URL, host.FOLDERS = data, own_url, folders


LORA_SUB = "LoRACasino"        # the folder inside ComfyUI's loras folder


TEMP_SUB = "_temp"             # LoRACasino/_temp: the throwaway LoRAs of each step


KEEP_TEMP = 40


QUALITY = {"breed": {"width": 512, "height": 688, "steps": 12}, "full": {"width": 768, "height": 1024, "steps": 20}}


DRAW = {"negative": "low quality, worst quality, blurry", "cfg": 5.0, "sampler": "dpmpp_2m", "scheduler": "karras", "steps": 12}


# example prompts in plain words, so they read on tag-trained and caption-trained checkpoints alike; yours go beside them
PROMPTS = [
    {"name": "Woman in a coat", "text": "a woman with glasses and brown braided hair wearing a trench coat and a scarf, upper body, looking at viewer, park with trees, daylight"},
    {"name": "Knight", "text": "a smiling knight in plate armor standing in a library, bookshelves, warm light"},
    {"name": "Mountain lake", "text": "a mountain lake at sunrise, pine forest, mist over the water, scenery, no people"},
    {"name": "Neon street", "text": "a rainy city street at night, neon signs, reflections on wet asphalt, scenery"},
    {"name": "Still life", "text": "a still life of fruit and a glass bottle on a wooden table, light from a window"},
]


_jobs = {}


class Reading(RuntimeError):
    """The checkpoint's numbers are still being worked out."""


NAME_OK = re.compile(r"^[\w.+\- ]{1,60}$")


def setup(body):
    """Which checkpoint, VAE and drawing settings a request uses, checked. 'tag' names them in picture file names."""
    name = str(body.get("ckpt") or "")
    path = ckpt_path(name)
    weights.layers(path)  # ValueError if it is not SDXL
    vae = body.get("vae")
    vae = "" if vae in (None, "", "checkpoint") else str(vae)
    neg = str(body.get("negative", DRAW["negative"]))[:2000]
    try:
        cfg = min(30.0, max(1.0, float(body.get("cfg", DRAW["cfg"]))))
    except (TypeError, ValueError):
        cfg = DRAW["cfg"]
    sampler, scheduler = str(body.get("sampler") or DRAW["sampler"]), str(body.get("scheduler") or DRAW["scheduler"])
    if not NAME_OK.match(sampler) or not NAME_OK.match(scheduler):
        raise ValueError("Unknown sampler or scheduler")
    tag = hashlib.sha1(f"{name}|{os.path.getsize(path)}|{vae}|{neg}|{cfg:g}|{sampler}|{scheduler}".encode()).hexdigest()[:8]
    return {"ckpt": name, "path": path, "vae": vae, "negative": neg, "cfg": cfg, "sampler": sampler, "scheduler": scheduler, "tag": tag}


def numbers(M):
    """The checkpoint's numbers, or Reading if they are still being worked out (the page waits on /api/model)."""
    St = weights.stats(M["path"])
    if St is None:
        s = weights.state(M["path"])
        St = weights.stats(M["path"])
        if St is None:
            raise Reading(s["error"] or "reading the checkpoint")
    return St


def size_from(body, base):
    """The size the page asked for ({"size": {"w", "h", "steps"}}), checked; else the default `base`.
    Sides are rounded to multiples of 16 (SDXL works in 8-pixel latents; 16 keeps both halves even), 256-2048,
    at most 2.4 megapixels."""
    d = body.get("size")
    b = QUALITY[base]
    if not isinstance(d, dict):
        return {"width": b["width"], "height": b["height"], "steps": b["steps"], "tag": f"{b['width']}x{b['height']}x{b['steps']}"}
    try:
        w, h = int(round(float(d.get("w")) / 16) * 16), int(round(float(d.get("h")) / 16) * 16)
        steps = int(d.get("steps") or b["steps"])
    except (TypeError, ValueError):
        raise ValueError("picture size must be numbers") from None
    w, h, steps = min(2048, max(256, w)), min(2048, max(256, h)), min(80, max(1, steps))
    if w * h > 2_400_000:
        raise ValueError(f"{w}×{h} is too big (over 2.4 megapixels); SDXL starts repeating things well before that")
    return {"width": w, "height": h, "steps": steps, "tag": f"{w}x{h}x{steps}"}


def workflow(prompt, seed, s, M, lora_name=None):
    model = ["1", 0]
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": M["ckpt"]}}}
    if lora_name:
        g["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": 1.0}}
        model = ["2", 0]
    g.update({
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": M["negative"]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": s["width"], "height": s["height"], "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
                                                   "seed": int(seed), "steps": s["steps"], "cfg": M["cfg"], "sampler_name": M["sampler"],
                                                   "scheduler": M["scheduler"], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["7", 0] if M["vae"] else ["1", 2]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    })
    if M["vae"]:
        g["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": M["vae"]}}
    return g


def temp_lora(St, mix, M):
    """Write (once) the throwaway LoRA for a mix on model M; -> its ComfyUI lora_name."""
    name = f"c_{mix_key(St, mix)}_{M['tag']}.safetensors"
    lf = lora_dir() / LORA_SUB / TEMP_SUB / name
    if not lf.exists():
        write_lora(lf, St, mix, M["path"])
    return f"{LORA_SUB}{os.sep}{TEMP_SUB}{os.sep}{name}"


def tidy_temp():
    try:
        d = lora_dir() / LORA_SUB / TEMP_SUB
    except ComfyDown:
        return
    files = sorted(d.glob("c_*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[KEEP_TEMP:]:
        try:
            p.unlink()
        except OSError:
            pass


def card_key(St, mix, prompt, seed, q, M):
    if not mix:
        return "base_" + hashlib.sha1(f"{prompt}|{seed}|{q['tag']}|{M['tag']}".encode()).hexdigest()[:12]
    return "push_" + hashlib.sha1(f"{mix_key(St, mix)}|{prompt}|{seed}|{q['tag']}|{M['tag']}".encode()).hexdigest()[:12]


def fetch_picture(pid, key, timeout=20):
    """If ComfyUI has finished prompt pid, save its picture as data/renders/<key>.png. -> True, False (not yet) or
    raises RuntimeError with ComfyUI's own error."""
    h = comfy(f"/history/{pid}", timeout=timeout)
    if pid not in h:
        return False
    st = h[pid].get("status", {})
    if st.get("status_str") == "error":
        msg = [m for m in st.get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError("ComfyUI: " + (msg[0][1].get("exception_message", "").strip()[:300] if msg else "the picture failed"))
    imgs = h[pid].get("outputs", {}).get("9", {}).get("images", [])
    if not imgs:
        return False
    im = imgs[0]
    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "temp")})
    RENDERS.mkdir(parents=True, exist_ok=True)
    tmp = RENDERS / f"{key}.{uuid.uuid4().hex[:6]}.tmp"
    tmp.write_bytes(comfy(f"/view?{q}", timeout=60))
    try:
        os.replace(tmp, RENDERS / f"{key}.png")
    except OSError:
        tmp.unlink(missing_ok=True)
    return (RENDERS / f"{key}.png").exists()


def queue(graph):
    try:
        r = comfy("/prompt", {"prompt": graph, "client_id": CID, "extra_data": PREVIEW})
    except urllib.error.HTTPError as e:  # ComfyUI answers 400 with what is wrong with the graph
        try:
            j = json.loads(e.read())
            why = (j.get("error") or {}).get("message", "") + " " + json.dumps(j.get("node_errors", {}))[:300]
        except Exception:  # noqa: BLE001
            why = str(e)
        raise RuntimeError(f"ComfyUI refused the picture: {why.strip()}") from None
    if r.get("node_errors"):
        raise RuntimeError(f"ComfyUI refused the picture: {json.dumps(r['node_errors'])[:300]}")
    return r["prompt_id"]


def render(graph, key, job=None):
    """Queue a graph, wait for it, save the picture as data/renders/<key>.png."""
    if (RENDERS / f"{key}.png").exists():
        return key
    live_start()
    pid = queue(graph)
    if job is not None:
        job["pid"] = pid
    t0 = time.time()
    while time.time() - t0 < 900:
        if fetch_picture(pid, key):
            return key
        time.sleep(0.4)
    raise TimeoutError("ComfyUI did not finish within 15 minutes")


def generate(job, St, mix, prompt, seed, q, M):
    try:
        t0 = time.time()
        bkey = card_key(St, [], prompt, seed, q, M)
        job.update(state="drawing the plain picture")
        render(workflow(prompt, seed, q, M), bkey, job)
        job.update(base=bkey, pid=None)
        if mix:
            lora = temp_lora(St, mix, M)
            tidy_temp()
            job.update(state="drawing the pushed picture")
            pkey = card_key(St, mix, prompt, seed, q, M)
            render(workflow(prompt, seed, q, M, lora), pkey, job)
            job.update(push=pkey)
        job.update(state="done", secs=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        job.update(state="error", error=str(e))


_rounds = {}


_round_lock = threading.Lock()


def cancel_round(r):
    """Drop a round's pictures that ComfyUI has not started yet (and stop the one it is drawing, if it is ours)."""
    r["cancelled"] = True
    pids = [c["pid"] for c in r["cards"] if c.get("pid") and not c.get("ready")]
    if not pids:
        return
    try:
        comfy("/queue", {"delete": pids})
        q = comfy("/queue")
        running = [x[1] for x in q.get("queue_running", [])]
        if any(p in running for p in pids):
            comfy("/interrupt", {})
    except Exception:  # noqa: BLE001
        pass


def run_round(r):
    St, M = r["stats"], r["model"]
    try:
        for c in r["cards"]:  # queue everything first so ComfyUI draws them back to back
            if r["cancelled"]:
                return
            if (RENDERS / f"{c['key']}.png").exists():
                c["ready"] = True
                continue
            lora = temp_lora(St, c["mix"], M) if c["mix"] else None
            c["pid"] = queue(workflow(r["prompt"], r["seed"], r["q"], M, lora))
        tidy_temp()
    except Exception as e:  # noqa: BLE001  (could not even queue: ComfyUI down or refused)
        r["error"] = str(e)
        return
    t0, fails = time.time(), 0
    while not r["cancelled"] and time.time() - t0 < 1800:
        todo = [c for c in r["cards"] if c.get("pid") and not c.get("ready") and not c.get("error")]
        if not todo:
            break
        for c in todo:
            if (RENDERS / f"{c['key']}.png").exists():  # another round already drew this one
                c["ready"] = True
                continue
            try:  # one slow or failed reply must not stop the rest of the round
                c["ready"] = bool(fetch_picture(c["pid"], c["key"]))
                fails = 0
            except RuntimeError as e:
                c["error"] = str(e)
            except Exception as e:  # noqa: BLE001
                fails += 1
                if fails > 40:
                    r["error"] = f"lost contact with ComfyUI: {e}"
                    return
                time.sleep(1)
        time.sleep(0.35)


def breed(body):
    prompt = str(body.get("prompt", "")).strip()[:2000]
    if not prompt:
        raise ValueError("Write a prompt first")
    M = setup(body)
    St = numbers(M)
    parent = clean_mix(body.get("parent"), St["n"])
    seed = int(body.get("seed", 6100)) % (2 ** 48)
    wild = min(1.0, max(0.0, float(body.get("wild", 0.5))))
    kn = knobs(body)
    n = 4 if int(body.get("n", 8)) == 4 else 8
    q = size_from(body, "breed")
    rng = np.random.default_rng()
    cid = str(body.get("client", ""))[:40]
    with _round_lock:
        for old in list(_rounds.values()):  # only this tab's unfinished step: two tabs share the queue instead
            if not old["cancelled"] and old.get("client") == cid:
                cancel_round(old)
        if len(_rounds) > 50:
            for k in sorted(_rounds, key=lambda k: _rounds[k]["t"])[:20]:
                _rounds.pop(k, None)
    children, seen = [], set()
    pk = card_key(St, parent, prompt, seed, q, M)
    seen.add(pk)
    if body.get("polish") and not parent:
        raise ValueError("Take a card first, then refine its hand.")
    if body.get("polish"):
        fine = min(1.0, max(0.0, float(body.get("fine", 0.3))))
        for mix, caps, move in polish(St, parent, n * 2, rng, fine):
            k = card_key(St, mix, prompt, seed, q, M)
            if mix and k not in seen and len(children) < n:
                seen.add(k)
                children.append({"key": k, "mix": mix, "changes": caps, "move": move})
    moves = deck(n, move_weights(wild, len(parent), sum(1 for m in parent if not m.get("held"))), rng)
    tries = 0 if not body.get("polish") else 80  # (polish fills the step itself)
    while len(children) < n and tries < 80:
        move = moves[len(children)] if tries < 40 else str(rng.choice(["leap", "add"]))
        tries += 1
        mix, caps = mutate(St, parent, move, rng, wild, kn)
        if not mix:
            continue
        k = card_key(St, mix, prompt, seed, q, M)
        if k in seen:
            continue
        seen.add(k)
        children.append({"key": k, "mix": mix, "changes": caps, "move": move})
    centre = {"key": pk, "mix": parent, "changes": [], "parent": True, "move": ""}
    cards = [centre] + children  # the picture you are at is drawn first (usually it already is)
    for c in cards:
        c["inside"] = [describe(St, m) for m in c["mix"]]
    live_start()
    rid = uuid.uuid4().hex[:10]
    r = {"id": rid, "t": time.time(), "prompt": prompt, "seed": seed, "q": q, "cards": cards, "cancelled": False, "model": M,
         "stats": St, "client": cid}
    _rounds[rid] = r
    threading.Thread(target=run_round, args=(r,), daemon=True).start()
    out = [{k: c[k] for k in ("key", "mix", "changes", "inside", "move")} | {"parent": c.get("parent", False)} for c in cards]
    return {"round": rid, "parent": out[0], "children": out[1:]}


def safe_name(s):
    return re.sub(r"[^\w.+-]+", "_", s).strip("._")[:70] or "look"


def save(St, mix, name, prompt, seed, rkey, M, qs):
    name = "casino_" + safe_name(name)
    lf = SAVED / f"{name}.safetensors"
    lora_name = f"{LORA_SUB}{os.sep}{lf.name}"
    rec = {"name": name, "made_by": "LoRA Casino", "made": time.strftime("%Y-%m-%d %H:%M:%S"),
           "mix": [{k: m[k] for k in ("label", "kind", "s")} | {"neurons": len(m["ids"])} for m in mix],
           "recipe": recipe_text(St, mix),
           "checkpoint": M["ckpt"], "vae": M["vae"] or "(the checkpoint's own)", "prompt": prompt, "negative": M["negative"], "seed": seed,
           "cfg": M["cfg"], "sampler": M["sampler"], "scheduler": M["scheduler"],
           "width": qs["width"], "height": qs["height"], "steps": qs["steps"],
           "lora": {"file": lf.name, "comfyui_lora_name": lora_name, "strength": 1.0,
                    "how": "LoraLoaderModelOnly (or LoraLoader). Strength 1 = the picture; it is linear: 0.5 half, 2 double, "
                           "-1 the opposite. Several of these LoRAs together add up.",
                    "made_for": f"{M['ckpt']} (the bias change uses this checkpoint's weights: exact on it, close on its relatives)"}}
    rec["layers"] = write_lora(lf, St, mix, M["path"], {"modelspec.title": name, "loracasino_recipe": json.dumps(rec)})
    pic = RENDERS / f"{rkey}.png" if rkey and re.match(r"^push_\w+$", rkey) else None
    if pic and pic.exists():
        (SAVED / f"{name}.preview.png").write_bytes(png_with_text(pic.read_bytes(), {
            "prompt": json.dumps(workflow(prompt, seed, qs, M, lora_name)), "loracasino": json.dumps(rec)}))
    (SAVED / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    dst = lora_dir() / LORA_SUB
    dst.mkdir(parents=True, exist_ok=True)
    for ext in (".safetensors", ".preview.png", ".json"):
        if (SAVED / f"{name}{ext}").exists():
            shutil.copyfile(SAVED / f"{name}{ext}", dst / f"{name}{ext}")
    return {"name": name, "file": str(lf), "comfyui": str(dst / lf.name), "lora_name": lora_name}


def handle(path, query):
    """GET -> (code, body, content type) or None if the path is not ours."""
    q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
    if path in ("/", "/index.html"):
        return 200, (PAGE / "index.html").read_bytes(), "text/html; charset=utf-8"
    if not path.startswith("/api/"):
        return None
    what = path[len("/api/"):]
    m = re.match(r"^render/((?:base|push)_\w{12})\.png$", what)
    if m:
        f = RENDERS / f"{m.group(1)}.png"
        return (200, f.read_bytes(), "image/png") if f.exists() else (404, {"error": "no picture"}, None)
    if what == "job":
        j = _jobs.get(q.get("id", ""))
        if not j:
            return 404, {"error": "no such job"}, None
        p, pv = live(j.get("pid"))
        return 200, dict(j) | {"p": p, "pv": pv, "live": _live["up"]}, None
    if what == "round":
        r = _rounds.get(q.get("id", ""))
        if not r:
            return 404, {"error": "no such round"}, None
        out = []
        for c in r["cards"]:
            p, pv = live(c.get("pid"))
            out.append({"key": c["key"], "ready": bool(c.get("ready")), "error": c.get("error"), "pid": c.get("pid"), "p": p, "pv": pv})
        return 200, {"cards": out, "error": r.get("error"), "cancelled": r["cancelled"], "live": _live["up"]}, None
    if what == "preview":  # the latest step sketch of a picture being drawn (the URL carries a counter, so caching is fine)
        b = _live["prev"].get(q.get("pid", ""))
        return (200, b, "image/jpeg") if b else (404, {"error": "no preview"}, None)
    if what == "meta":
        ok = find_comfy() is not None
        return 200, {"comfy": ok, "comfy_url": host.COMFY if ok else "", "comfy_tried": comfy_try(), "comfy_set": settings().get("comfy", ""),
                     "embedded": bool(host.OWN_URL),
                     "prompts": PROMPTS, "knobs": KNOBS, "draw": DRAW}, None
    if what == "models":
        try:
            return 200, list_models(), None
        except ComfyDown as e:
            return 503, {"error": str(e)}, None
    if what == "model":  # is this checkpoint usable, and are its numbers ready (reading starts here)
        try:
            p = ckpt_path(q.get("ckpt", ""))
            return 200, {"ok": True} | weights.state(p), None
        except ComfyDown as e:
            return 503, {"error": str(e)}, None
        except (ValueError, OSError, KeyError) as e:
            return 200, {"ok": False, "error": str(e)}, None
    return 404, {"error": "unknown"}, None


def post(path, body):
    """POST -> (code, body)."""
    try:
        if path == "/api/settings":
            set_settings(body)
            ok = find_comfy() is not None
            return 200, {"ok": True, "comfy": ok, "comfy_url": host.COMFY if ok else "", "comfy_tried": comfy_try()}
        if path == "/api/breed/stop":
            r = _rounds.get(str(body.get("round", "")))
            if r and not r["cancelled"]:
                cancel_round(r)
            return 200, {"ok": True}
        if path == "/api/breed":
            return 200, breed(body)
        if path not in ("/api/generate", "/api/save"):
            return 404, {"error": "not found"}
        prompt = str(body.get("prompt", "")).strip()[:2000]
        seed = int(body.get("seed", 6100)) % (2 ** 48)
        M = setup(body)
        St = numbers(M)
        mix = clean_mix(body.get("mix"), St["n"])
        q = size_from(body, "full")
        if path == "/api/generate":
            if not prompt:
                return 400, {"error": "Write a prompt first"}
            jid = uuid.uuid4().hex[:10]
            _jobs[jid] = {"id": jid, "state": "queued", "t": time.time()}
            if len(_jobs) > 200:
                for k in sorted(_jobs, key=lambda k: _jobs[k]["t"])[:50]:
                    _jobs.pop(k, None)
            threading.Thread(target=generate, args=(_jobs[jid], St, mix, prompt, seed, q, M), daemon=True).start()
            return 200, {"job": jid}
        if not mix:
            return 400, {"error": "Nothing is pushed in this one"}
        return 200, save(St, mix, str(body.get("name", "")), prompt, seed, str(body.get("render", "")), M, q)
    except Reading as e:
        return 409, {"error": str(e), "reading": True}
    except ComfyDown as e:
        return 503, {"error": str(e)}
    except (ValueError, TypeError) as e:
        return 400, {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return 500, {"error": str(e) or type(e).__name__}
