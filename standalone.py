"""LoRA Casino on its own, beside ComfyUI (the other way to run it is as a ComfyUI plugin: see README).

    python standalone.py [PORT] [--open]        then open http://localhost:8777

"Start LoRA Casino.bat" (Windows) and "start.sh" (Mac, Linux) do this for you. It listens on this PC only, and
ComfyUI has to run on the same PC: LoRA Casino reads the checkpoint file and writes LoRAs into ComfyUI's folder.
Needs Python 3.10+ and numpy."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from casino import app  # noqa: E402

app.configure(ROOT / "data", ROOT / "page")
OPEN = "--open" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--open"]
PORT = int(ARGS[0]) if ARGS else 8777


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json; charset=utf-8", cache=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache or "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, OSError):  # the page went away mid-reply
            pass

    def do_GET(self):
        u = urlparse(self.path)
        try:
            r = app.handle(unquote(u.path), u.query)
        except Exception as e:  # noqa: BLE001
            return self.reply(500, {"error": str(e) or type(e).__name__})
        if r is None:
            return self.reply(404, {"error": "not found"})
        code, body, ctype = r
        return self.reply(code, body, ctype or "application/json; charset=utf-8",
                          "max-age=3600" if ctype in ("image/jpeg", "image/png") else None)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            return self.reply(400, {"error": "bad request"})
        code, out = app.post(urlparse(self.path).path, body)
        return self.reply(code, out)


if __name__ == "__main__":
    import urllib.request
    import webbrowser

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False  # on Windows reuse would let a second copy silently share the port
        daemon_threads = True

    url = f"http://localhost:{PORT}"
    try:
        server = Server(("127.0.0.1", PORT), Handler)
    except OSError:  # port taken: if it is already LoRA Casino, just open it
        try:
            meta = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/meta", timeout=5).read())
            assert "knobs" in meta
            print(f"LoRA Casino is already running on {url} - opening it.")
            if OPEN:
                webbrowser.open(url)
            sys.exit(0)
        except Exception:  # noqa: BLE001
            print(f"Port {PORT} is used by another program. Close it, or run: standalone.py {PORT + 1} --open")
            sys.exit(1)
    print(f"LoRA Casino on {url}", flush=True)
    print("Keep this window open while you play; close it to stop.", flush=True)
    if OPEN:
        webbrowser.open(url)  # only now: the server is already listening, so the page loads
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
