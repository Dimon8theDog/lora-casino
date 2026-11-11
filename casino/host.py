"""The only part of LoRA Casino that talks to ComfyUI.

Everything goes through ComfyUI's public HTTP API and its websocket: the model lists, queueing a picture, its progress
and step sketches, fetching the result. That is the same whether LoRA Casino runs inside ComfyUI (as a plugin) or on
its own beside it; the two differ only in how ComfyUI is found:

  OWN_URL   plugin: a function returning ComfyUI's own address. Standalone: None, and ComfyUI is looked for at the
            address set in the page, then on its usual ports.
  FOLDERS   plugin: a function kind -> folders (ComfyUI's folder_paths). Standalone: None, and ComfyUI is asked over HTTP.
  DATA      where settings are kept.
"""
import json
import os
import re
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OWN_URL = None
FOLDERS = None
DATA = None


def settings_file():
    return Path(DATA) / "settings.json"


def _tls(url):
    """ComfyUI started with its own certificate is still this PC talking to itself: don't ask who signed it."""
    return ssl._create_unverified_context() if str(url).startswith("https") else None


COMFY_USUAL = ["http://127.0.0.1:8188", "http://127.0.0.1:8000"]


COMFY = COMFY_USUAL[0]


def settings():
    try:
        s = json.loads(settings_file().read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def set_settings(body):
    s = settings()
    if "comfy" in body:
        u = str(body.get("comfy") or "").strip().rstrip("/")
        if u and not re.match(r"^https?://[\w.\-\[\]:]+(:\d+)?(/[\w./-]*)?$", u):
            raise ValueError("The ComfyUI address should look like http://127.0.0.1:8188")
        s["comfy"] = u
    Path(DATA).mkdir(parents=True, exist_ok=True)
    tmp = settings_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    os.replace(tmp, settings_file())
    _folders.update(t=0.0, d={})
    return s


def comfy_try():
    if OWN_URL:  # inside ComfyUI: it is the one we run in
        return [OWN_URL()]
    mine = settings().get("comfy") or os.environ.get("CASINO_COMFY_URL")
    return ([mine] if mine else []) + [u for u in COMFY_USUAL if u != mine]


class ComfyDown(RuntimeError):
    pass


def find_comfy():
    """The first ComfyUI that answers: the address set in the page, then the usual ports 8188 and 8000."""
    global COMFY
    import socket
    tries = comfy_try()
    for u in ([COMFY] if COMFY in tries else []) + [x for x in tries if x != COMFY]:
        try:
            p = urllib.parse.urlparse(u)  # a closed port on Windows takes 2 s to refuse an HTTP request: knock first
            socket.create_connection((p.hostname or "127.0.0.1", p.port or (443 if p.scheme == "https" else 80)), timeout=0.35).close()
            with urllib.request.urlopen(u + "/system_stats", timeout=2, context=_tls(u)) as r:
                r.read()
            COMFY = u
            return u
        except Exception:  # noqa: BLE001
            continue
    return None


def comfy(path, body=None, timeout=10, again=True):
    req = urllib.request.Request(COMFY + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_tls(COMFY)) as r:
            raw = r.read()
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, OSError):
        if not again or find_comfy() is None:
            raise ComfyDown(f"ComfyUI isn't running (tried {', '.join(comfy_try())}). Start it and draw again.") from None
        return comfy(path, body, timeout, False)
    return raw if path.startswith("/view") else json.loads(raw)


# live progress (ComfyUI's websocket): our pictures are queued with this client id and a latent preview per step
# ("latent2rgb", set per prompt; the user's own ComfyUI setting is untouched). A small stdlib websocket client listens
# for "progress" (step k of n) and the preview frames, so the page can show each picture being drawn.
CID = "lora-casino"


PREVIEW = {"preview_method": "latent2rgb"}


_live = {"up": False, "pid": None, "prog": {}, "prev": {}, "n": {}, "started": False}


def _trim(d, keep=120):
    for k in list(d)[: max(0, len(d) - keep)]:
        d.pop(k, None)


def _on_ws(op, pay):
    if op == 1:
        try:
            m = json.loads(pay)
        except ValueError:
            return
        t, d = m.get("type"), m.get("data") or {}
        pid = d.get("prompt_id") if isinstance(d, dict) else None
        if not pid:
            return
        if t == "execution_start":
            _live["pid"] = pid
        elif t == "progress":
            _live["pid"] = pid
            _live["prog"][pid] = (int(d.get("value", 0)), max(1, int(d.get("max", 1))))
            _trim(_live["prog"])
        elif t == "executing":
            _live["pid"] = pid if d.get("node") is not None else None
    elif op == 2 and len(pay) > 8 and struct.unpack(">I", pay[:4])[0] == 1 and _live["pid"]:
        pid = _live["pid"]  # PREVIEW_IMAGE: event, image type, then the JPEG
        _live["prev"][pid] = bytes(pay[8:])
        _live["n"][pid] = _live["n"].get(pid, 0) + 1
        _trim(_live["prev"], 40)
        _trim(_live["n"])


def _ws_frame(op, payload):
    mask = os.urandom(4)
    head = bytes([0x80 | op, 0x80 | len(payload)]) + mask
    return head + bytes(c ^ mask[i % 4] for i, c in enumerate(payload))


def _ws_loop():
    import base64
    import socket
    while True:
        s = None
        try:
            u = urllib.parse.urlparse(COMFY)
            host, port = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80)
            s = socket.create_connection((host, port), timeout=5)
            if u.scheme == "https":
                s = _tls(COMFY).wrap_socket(s, server_hostname=host)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall((f"GET /ws?clientId={CID} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                ch = s.recv(4096)
                if not ch:
                    raise ConnectionError("closed")
                buf += ch
            head, buf = buf.split(b"\r\n\r\n", 1)
            if b" 101" not in head.split(b"\r\n", 1)[0]:
                raise ConnectionError("ComfyUI refused the websocket")
            s.settimeout(60)
            _live["up"] = True

            def need(n):
                nonlocal buf
                while len(buf) < n:
                    try:
                        ch = s.recv(65536)
                    except socket.timeout:
                        continue  # quiet while nothing is drawing
                    if not ch:
                        raise ConnectionError("closed")
                    buf += ch
                out, buf = buf[:n], buf[n:]
                return out

            frag, fop = b"", 0
            while True:
                b0, b1 = need(2)
                op, ln = b0 & 0x0F, b1 & 0x7F
                if ln == 126:
                    ln = struct.unpack(">H", need(2))[0]
                elif ln == 127:
                    ln = struct.unpack(">Q", need(8))[0]
                mask = need(4) if b1 & 0x80 else None
                pay = need(ln)
                if mask:
                    pay = bytes(c ^ mask[i % 4] for i, c in enumerate(pay))
                if op == 8:
                    raise ConnectionError("closed by ComfyUI")
                if op == 9:
                    s.sendall(_ws_frame(0xA, pay[:125]))
                    continue
                if op == 0:
                    frag += pay
                elif op in (1, 2):
                    frag, fop = pay, op
                else:
                    continue
                if b0 & 0x80:
                    _on_ws(fop, frag)
                    frag = b""
        except Exception:  # noqa: BLE001  (ComfyUI down or restarting: try again in a moment)
            _live["up"] = False
            time.sleep(3)
        finally:
            if s:
                try:
                    s.close()
                except OSError:
                    pass


def live_start():
    if not _live["started"]:
        _live["started"] = True
        threading.Thread(target=_ws_loop, daemon=True, name="comfy-ws").start()


def live(pid):
    """Progress of one queued picture: fraction of sampling steps done (None = not started) and preview count."""
    if not pid:
        return None, 0
    v = _live["prog"].get(pid)
    return (round(v[0] / v[1], 3) if v else None), _live["n"].get(pid, 0)


_folders = {"t": 0.0, "d": {}}


_lists = {"t": 0.0, "d": None}


def comfy_folders():
    """Every folder ComfyUI loads each kind of model from, asked from ComfyUI itself; kept for 30 s."""
    if _folders["d"] and time.time() - _folders["t"] < 30:
        return _folders["d"]
    d = {}
    if FOLDERS:  # inside ComfyUI: it tells us directly
        d = {k: [Path(p) for p in FOLDERS(k)] for k in ("checkpoints", "vae", "loras")}
    elif find_comfy():
        try:
            for f in comfy("/experiment/models", timeout=5):
                d[f["name"]] = [Path(p) for p in f.get("folders", [])]
        except Exception:  # noqa: BLE001
            d = {}
    _folders.update(t=time.time(), d=d)
    return d


def model_roots(kind):
    found, seen = [], set()
    for r in comfy_folders().get(kind, []):
        k = str(r).lower()
        if k not in seen and r.exists():
            seen.add(k)
            found.append(r)
    return found


def lora_dir():
    roots = model_roots("loras")
    if not roots:
        raise ComfyDown("ComfyUI didn't say where its loras folder is. LoRA Casino has to run on the same PC as ComfyUI.")
    return roots[0]


def list_models():
    """What ComfyUI can load and draw with: checkpoints, VAEs, samplers, schedulers; kept for 20 s."""
    if _lists["d"] and time.time() - _lists["t"] < 20:
        return _lists["d"]
    if not find_comfy():
        raise ComfyDown(f"ComfyUI isn't running (tried {', '.join(comfy_try())}).")
    out = {}
    for kind in ("checkpoints", "vae"):
        names = comfy(f"/models/{kind}", timeout=8)
        out[kind] = [n for n in names if isinstance(n, str) and n.lower().endswith(".safetensors")]
    try:
        k = comfy("/object_info/KSampler", timeout=8)["KSampler"]["input"]["required"]

        def options(spec):  # ["a", "b"] in older ComfyUI, ["COMBO", {"options": [...]}] in newer
            return spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
        out["samplers"], out["schedulers"] = options(k["sampler_name"]), options(k["scheduler"])
    except Exception:  # noqa: BLE001
        out["samplers"], out["schedulers"] = ["dpmpp_2m"], ["karras"]
    _lists.update(t=time.time(), d=out)
    return out


def ckpt_path(name):
    if not name:
        raise ValueError("Pick a checkpoint first (Tune, under Model)")
    for r in model_roots("checkpoints"):
        p = (r / name).resolve()
        if p.suffix == ".safetensors" and p.exists() and r.resolve() in p.parents:
            return p
    if not comfy_folders():
        raise ComfyDown(f"ComfyUI isn't running (tried {', '.join(comfy_try())}). Start it and draw again.")
    raise ValueError(f"Checkpoint {name} wasn't found in ComfyUI's checkpoint folders")
