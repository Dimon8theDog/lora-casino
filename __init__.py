"""LoRA Casino as a ComfyUI plugin: this folder sits in ComfyUI/custom_nodes and ComfyUI loads this file at start.

It adds the table at  <ComfyUI address>/lora-casino/  (and a button for it in ComfyUI's top bar, see web/), served by
ComfyUI's own server. No nodes are added to the graph. ComfyUI's Python already has what it needs (numpy).

Everything here is the thin plugin side; the game is in casino/ and is the same code standalone.py runs."""
import asyncio
import logging
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

BASE = "/lora-casino"
HERE = Path(__file__).resolve().parent


def _own_url():
    """ComfyUI's own address, for the requests LoRA Casino makes to its public API."""
    from comfy.cli_args import args
    from server import PromptServer
    s = PromptServer.instance
    addr = str(getattr(s, "address", "") or "127.0.0.1")
    if addr in ("0.0.0.0", "::", "", "localhost"):  # listening on every address: reach it on this PC
        addr = "127.0.0.1"
    if ":" in addr:
        addr = f"[{addr}]"
    scheme = "https" if getattr(args, "tls_keyfile", None) and getattr(args, "tls_certfile", None) else "http"
    return f"{scheme}://{addr}:{getattr(s, 'port', None) or 8188}"


def _mount():
    import folder_paths
    from aiohttp import web
    from server import PromptServer

    from .casino import app

    data = Path(folder_paths.get_user_directory()) / "lora-casino"  # pictures, saved looks, per-checkpoint numbers
    app.configure(data, HERE / "page", own_url=_own_url, folders=folder_paths.get_folder_paths)
    routes = PromptServer.instance.routes

    def reply(code, body, ctype=None):
        if isinstance(body, (dict, list)):
            return web.json_response(body, status=code, headers={"Cache-Control": "no-store"})
        cache = "max-age=3600" if ctype in ("image/jpeg", "image/png") else "no-store"
        return web.Response(body=body, status=code, content_type=(ctype or "application/octet-stream").split(";")[0],
                            charset="utf-8" if ctype and "charset" in ctype else None, headers={"Cache-Control": cache})

    @routes.get(BASE)
    async def _to_page(request):  # the page asks for "api/..." beside itself, so it has to end in a slash
        raise web.HTTPFound(BASE + "/")

    @routes.get(BASE + "/")
    async def _page(request):
        return reply(*app.handle("/", ""))

    # the engine's work (reading a checkpoint, asking ComfyUI) runs off ComfyUI's event loop, which stays free
    @routes.get(BASE + "/api/{what:.*}")
    async def _get(request):
        try:
            r = await asyncio.to_thread(app.handle, "/api/" + request.match_info["what"], request.query_string)
        except Exception as e:  # noqa: BLE001
            return reply(500, {"error": str(e) or type(e).__name__})
        return reply(*r) if r else reply(404, {"error": "not found"})

    @routes.post(BASE + "/api/{what:.*}")
    async def _post(request):
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:  # noqa: BLE001
            return reply(400, {"error": "bad request"})
        code, out = await asyncio.to_thread(app.post, "/api/" + request.match_info["what"], body)
        return reply(code, out)


try:
    _mount()
except Exception as e:  # noqa: BLE001  never stop ComfyUI from starting
    logging.getLogger("lora-casino").error("LoRA Casino could not start: %s", e, exc_info=True)
