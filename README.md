# LoRA Casino

Grow a look by picking the picture you like, then save it as a LoRA. Works with any SDXL checkpoint
(SDXL itself, Pony, Illustrious, NoobAI and their merges). No training, and no data that belongs to one checkpoint.

https://github.com/user-attachments/assets/ec3d6ebf-ee75-4648-bbec-839cacb9da8c

It plays like a card table. Each deal lays out the same prompt nine times (or four). Every picture plays a different
**hand**: a few **cards**, each of which pushes some of the model's neurons up or down. Take the picture you like and
the next deal grows from its hand. A hand you like cashes out as a small LoRA (about 30 kB) that ComfyUI loads like
any other.

| On the table | What it does |
| --- | --- |
| **Deal** | Lays out a new set of pictures around your card. Cards lie face down and turn over as ComfyUI draws them. |
| Take a card | Click a picture: it becomes your card, and the next deal starts from its hand. |
| **Your hand** | The cards behind your picture: where the neurons sit (layout, style, detail), how strong, up or down. Point at a dealt picture to see its hand and what differs from yours. |
| **Hold** / **Discard** | On each card in your hand. A held card is left alone by every deal; Discard drops the card and deals again. |
| **Stakes** | Low: a deal changes one card a little. High: remixes and whole new hands, and more pictures that break. |
| **Draw** / **Refine** | Draw deals new cards; Refine keeps the cards and changes them a little. |
| **Take back**, **Shuffle**, **Fold** | One card back; same hand with new starting noise; give up the hand. |
| **Winnings**, **Trail** | Pictures you kept with the chip; the cards you took, in order. |
| **Cash out as LoRA** | In a picture's close-up: saves its hand as a LoRA into ComfyUI. |

## Install

There are two ways to run it. They are the same game and the same code; pick one.

### Inside ComfyUI (a plugin)

1. Put this folder into ComfyUI's `custom_nodes` folder, named `lora-casino`:
   - with ComfyUI's Manager: *Install via Git URL*, and give it `https://github.com/Dimon8theDog/lora-casino`; or
   - by hand, inside `ComfyUI/custom_nodes`: `git clone https://github.com/Dimon8theDog/lora-casino`
2. Restart ComfyUI.
3. Press the playing-cards button in ComfyUI's top bar, or open `<ComfyUI's address>/lora-casino/`
   (usually `http://127.0.0.1:8188/lora-casino/`).

Nothing else to install or set: ComfyUI's Python already has what it needs, and LoRA Casino takes the model folders
from ComfyUI. It works wherever ComfyUI runs (Windows, Mac, Linux, portable, Desktop, a remote machine), because it
runs in the same place as the models. It adds no nodes to the graph. Your pictures, saved looks and the per-checkpoint
numbers go to `ComfyUI/user/lora-casino`, so updating the plugin never touches them.

### On its own, beside ComfyUI

1. Start ComfyUI on the same computer.
2. Windows: double-click **Start LoRA Casino.bat**. Mac / Linux: run `sh start.sh`. Or, anywhere:
   `python standalone.py --open`. The page opens at `http://localhost:8777`.

The launchers look for a Python 3.10+ with numpy (their own `.venv`, then the one on the PATH, and on Windows the
one ComfyUI Desktop ships). If they find a Python without numpy they make a `.venv` here once and install numpy into
it. ComfyUI is found on its usual ports (8188, 8000); if yours is elsewhere, set its address in the page under
**Table**. In this mode ComfyUI has to be on the same computer, because LoRA Casino reads the checkpoint file and
writes LoRAs into ComfyUI's folder. Data goes to `data/` in this folder.

### Either way

- Checkpoints must be `.safetensors` files in ComfyUI's checkpoint folders, and SDXL.
- It writes into ComfyUI's `loras/LoRACasino/` (looks you cashed out) and `loras/LoRACasino/_temp/` (the throwaway
  LoRAs each deal is drawn with; the newest 40 are kept).
- Checkpoint, VAE, negative prompt, steps, CFG, sampler and scheduler are set in the page, under **Table**.

## How it fits any checkpoint

The first time a checkpoint is used its weights are read once (about a minute) and the result is kept in
`data/cache`. From the weights alone it works out, for each of SDXL's 332,800 feed-forward neurons:

- **how big a push should be**: an estimate of the neuron's typical activation, from the layer's own LayerNorm and
  input weights;
- **its neighbours**: the neurons of the same layer that write in the most similar direction;
- **groups of about 1000 and pieces of about 100** neurons that write in similar directions.

A push is a change of one layer's bias (`W @ shift`), saved as standard `diff_b` LoRA keys, so a saved LoRA is exact
on the checkpoint it was made with and close on its relatives.

### What was checked, and what was not

- Checked against activation sizes measured on real pictures for one checkpoint (prefectIllustriousXL_40): the
  estimate has the right level per layer (within about ±40 %); single neurons inside a layer are noisier (about ×1.4).
- Drawn on three checkpoints (prefectIllustriousXL_40, prefect_illustrious_v8 fp16, Illustrious-XL-v0.1), with tag
  prompts and plain-word prompts, a person and a landscape.
- **Checkpoints differ in how hard they can be pushed.** The two merges gave whole pictures at the default strength;
  the Illustrious base broke about half of them at the same strength. If pictures break, lower *Card strength* in
  the Table panel (or lower the stakes). The weights do not tell how sturdy a checkpoint is.
- Both ways of running were played on one Windows PC with ComfyUI 0.42.1 (Desktop install): deal, take a card,
  cash out. Not tried: Mac and Linux (`start.sh` is written but has never been run), a remote ComfyUI, ComfyUI started
  with its own certificate, installing through the Manager, and the GitHub test run.
- Not tried: Pony, NoobAI, plain SDXL, few-step (Lightning / Turbo) checkpoints. All three tested are Illustrious
  family, because those are the SDXL checkpoints on this PC.
- Neighbours and groups come from write directions, not from what neurons fire on, so a card is named by where it
  sits (Style, Layout, Detail) and recognised by its emblem rather than by a tag.
- Tried and not used: reading a word for each neuron straight from the weights (word vector → the block's
  cross-attention → the neuron's input weights, the "logit lens" idea from language models). Scored against tags
  measured on 2000 pictures for prefectIllustriousXL_40, 306 tags: the top word from weights was among a neuron's
  five measured tags 4 % of the time (chance 1.7 %), 10 % in the deepest encoder block, and close to chance for whole
  cards. A real signal, far too weak to name cards with.

## Layout

| Path | What |
| --- | --- |
| `__init__.py` | The plugin side: ComfyUI loads it, and it puts the table on ComfyUI's own server under `/lora-casino/`. |
| `web/lora_casino.js` | The button and command in ComfyUI's interface. |
| `standalone.py`, `Start LoRA Casino.bat`, `start.sh` | The other way to run it: a small server of its own (this computer only, port 8777) and its launchers. |
| `casino/engine.py` | The game: hands, cards, the moves a deal makes, turning a hand into LoRA weights. No ComfyUI in it. |
| `casino/weights.py` | Reads a checkpoint and works out the numbers above. |
| `casino/host.py` | The only part that talks to ComfyUI, through ComfyUI's public HTTP API and websocket. |
| `casino/app.py` | Answers the page: deals a round, draws it, cashes a hand out. Used by both ways of running. |
| `tests/` | Engine tests on a tiny stand-in checkpoint (a few hundred kB with SDXL's layer names): `python -m unittest discover -s tests`. They need only numpy, no ComfyUI, GPU or model. `.github/workflows/tests.yml` runs them on Windows, Linux and Mac. |
| `pyproject.toml` | Name, version and repository address. Listing it in ComfyUI's registry would also need a publisher id. |
| `LICENSE` | MIT: anyone may use, change, share and sell it, as long as the licence text stays with it. |
| `page/index.html` | The page: one file, with its own icons drawn in it and the system's fonts, so it needs nothing from the internet. |
| `data/` | Made while running on its own (as a plugin: `ComfyUI/user/lora-casino`): `renders/` pictures, `saved/` LoRAs with preview and recipe, `cache/` per-checkpoint numbers, `settings.json`. Not part of the repository. |

Winnings, your prompts and the panel settings are kept by the browser.

## Where it came from

It grew out of a research tool that bred looks from a neuron map measured on one checkpoint. LoRA Casino keeps the
idea and drops everything tied to that checkpoint: the measured map (push sizes, groups, neighbours, tag names), the
fixed model, VAE and prompts. What it needs is worked out from whichever checkpoint you pick.
