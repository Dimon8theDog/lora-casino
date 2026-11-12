# LoRA Casino

Grow a look by picking the picture you like, then save it as a LoRA. Works with SDXL checkpoints
(SDXL itself, Pony, Illustrious, NoobAI and their merges) and with Anima. No training, and no data that belongs to
one checkpoint.

https://github.com/user-attachments/assets/ec3d6ebf-ee75-4648-bbec-839cacb9da8c

It plays like a card table. Each deal lays out the same prompt nine times (or four). Every picture plays a different
**hand**: a few **cards**, each of which pushes some of the model's neurons up or down. Take the picture you like and
the next deal grows from its hand. A hand you like cashes out as a small LoRA (tens of kB) that ComfyUI loads like
any other.

| On the table | What it does |
| --- | --- |
| **Deal** | Lays out a new set of pictures around your card. Cards lie face down and turn over as ComfyUI draws them. |
| Take a card | Click a picture: it becomes your card, and the next deal starts from its hand. |
| **Your hand** | The cards behind your picture: where the neurons sit (SDXL: layout, style, detail; Anima: prompt, layout, subject, rendering), how strong, up or down. Point at a dealt picture to see its hand and what differs from yours. |
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

- Models must be `.safetensors` files in ComfyUI's folders:
  - **SDXL**: a whole checkpoint, in `checkpoints`.
  - **Anima**: the model in `diffusion_models`, its text encoder (`qwen_3_06b_base`) in `text_encoders` and its VAE
    (`qwen_image_vae`) in `vae`, as ComfyUI's own Anima workflow has them. LoRA Casino picks those two by name; you
    can change them under **Table**.
- It writes into ComfyUI's `loras/LoRACasino/` (looks you cashed out) and `loras/LoRACasino/_temp/` (the throwaway
  LoRAs each deal is drawn with; the newest 40 are kept).
- Model, VAE, text encoder, negative prompt, steps, CFG, sampler and scheduler are set in the page, under **Table**.
  Each family starts with the settings its makers recommend and keeps its own: switching from SDXL to Anima and back
  does not mix them up. A hand belongs to one family, so switching family folds it.

## How it fits any checkpoint

The first time a checkpoint is used its weights are read once (about a minute) and the result is kept in
`data/cache`. From the weights alone it works out, for each feed-forward neuron (SDXL has 332,800 in 70 layers,
Anima 253,952 in 34):

- **how big a push should be**: an estimate of the neuron's typical activation, from the layer's own norm and
  input weights;
- **its neighbours**: the neurons of the same layer that write in the most similar direction;
- **groups of about 1000 and pieces of about 100** neurons that write in similar directions.

### SDXL

A push is a change of one layer's bias (`W @ shift`), saved as standard `diff_b` LoRA keys, so a saved LoRA is exact
on the checkpoint it was made with and close on its relatives. A card's strength is a multiple of each neuron's
typical activation.

### Anima

Anima is a transformer (28 blocks) with a small text adapter in front (6 blocks) that turns the prompt into what the
blocks read. Its 28 blocks have **no biases at all**, so the SDXL push has nothing to change there. Instead:

- **A card in a block** gives its neurons one shared push along the direction they write in on average. That is a
  rank-1 change of the block's output weight, saved as ordinary `lora_up` / `lora_down` keys (about 40 kB per card).
  For a single neuron it is exactly "this neuron, louder or quieter"; for a group it is the part the neurons agree
  on. In the first test this did about as much to the picture as scaling every neuron of the group on its own, which
  takes a LoRA a thousand times the size.
- **A card in the text adapter** (the Prompt suit) is the SDXL push unchanged, because the adapter does have biases.
  It changes how the prompt is read: the picture stays clean and drifts away from the prompt as the card gets
  stronger.
- **Suits** go by where a card sits: Prompt (the adapter), then Layout, Subject and Rendering (the first, middle and
  last third of the blocks). On the one checkpoint tried, an early block changed the pose and view, a middle block
  what is in the picture (clothes, composition), a late block how it is painted. Only three blocks were looked at
  one by one, so where one suit ends and the next begins is a guess.
- **Strength** counts the size of the card's whole push against the size of its layer's whole output. A block has
  no bias, but it shifts, scales and gates its feed-forward by the timestep, and those weights are in the file, so
  each neuron's usual activation can be estimated the same way as on SDXL. The point is that the same strength then
  means about the same on any card, whether it has one neuron or a thousand. Cards start at about 2.8 (Subject and
  Rendering a quarter lower, Prompt at 1), and a deal at the default settings plays them at about half of that.

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
  with its own certificate, and installing through the Manager.
- Not tried: Pony, NoobAI, plain SDXL, few-step (Lightning / Turbo) checkpoints. All three SDXL checkpoints tested
  are Illustrious family, because those are the SDXL checkpoints on this PC.

**Anima** was tried on one checkpoint (anima-aesthetic-v1.1), on the same PC:

- **Which push.** 34 pictures of one prompt at 1024×1024: scaling a group's neurons one by one (a 20 MB LoRA), the
  shared push used now (20 kB), and the bias push in the text adapter. All three gave whole, different pictures
  when gentle and broke when strong; the shared push moved the picture about as far as the exact scaling.
- **What strength counts.** With strength as a plain factor, cards of the same suit and size differed wildly: of 24
  cards, some broke the picture at a quarter of the starting strength and some did nothing at full. Counting the
  push against its layer's whole output (estimated from the weights) put the 144 pictures in a much better order
  (rank correlation 0.60 with how far the picture moved, 0.38 before), and every picture that came out flat or
  black had a strength above 1.9.
- **Checked on new cards.** 64 further cards on two prompts, sized that way (a picture counted as broken when it
  turned to noise, stripes or a flat wash): at 1.2, 7 of 96 block pictures broke (Layout 0, Subject 4, Rendering
  3); at 2.4, 22 of 96. Prompt cards moved the picture the same amount at every card size and broke none.
- **What is left over.** The estimate is rough. At the default settings about one block card in five barely
  changes the picture, and about one Subject or Rendering card in ten breaks it. The defaults are set for that: a
  new hand has one or two cards, mostly Layout and Subject. Two deals at the defaults (a landscape and a figure in
  a room) gave 16 pictures, 1 of them broken; a third deal, grown from a card taken from the second, gave 8 whole
  ones. If more break, lower *Card strength*.
- **Played through**, both ways of running: deal, take a card, deal from its hand, cash out. The cashed-out LoRA
  (46 kB, one card in a block and one in the text adapter), loaded on its own in ComfyUI with ComfyUI's Anima
  nodes, gave the card's picture exactly; at strength 0 it gave the plain picture.
- Reading the checkpoint takes about 70 seconds. A picture at 512×688 and 16 steps takes about 4 seconds on an
  RTX 3080 Ti, a deal of nine about 35.
- Not tried on Anima: any other Anima checkpoint (the base and preview releases), an Anima checkpoint that holds
  the text encoder and VAE in one file, the few-step turbo LoRA together with cards, and how a look made at 512×688
  holds up at 1024×1024.
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
| `casino/families.py` | What differs between kinds of model (SDXL, Anima): how to recognise the file, where the neurons are, what a push is there, the suits, the starting strengths, how it is drawn. A new kind of model starts here. |
| `casino/engine.py` | The game: hands, cards, the moves a deal makes, turning a hand into LoRA weights. No ComfyUI in it. |
| `casino/weights.py` | Reads a checkpoint and works out the numbers above. |
| `casino/host.py` | The only part that talks to ComfyUI, through ComfyUI's public HTTP API and websocket. |
| `casino/app.py` | Answers the page: deals a round, draws it, cashes a hand out. Used by both ways of running. |
| `tests/` | Engine tests on tiny stand-in files (a few hundred kB with SDXL's and Anima's layer names): `python -m unittest discover -s tests`. They need only numpy, no ComfyUI, GPU or model. `.github/workflows/tests.yml` runs them on Windows, Linux and Mac. |
| `pyproject.toml` | Name, version and repository address. Listing it in ComfyUI's registry would also need a publisher id. |
| `LICENSE` | MIT: anyone may use, change, share and sell it, as long as the licence text stays with it. |
| `page/index.html` | The page: one file, with its own icons drawn in it and the system's fonts, so it needs nothing from the internet. |
| `data/` | Made while running on its own (as a plugin: `ComfyUI/user/lora-casino`): `renders/` pictures, `saved/` LoRAs with preview and recipe, `cache/` per-checkpoint numbers, `settings.json`. Not part of the repository. |

Winnings, your prompts and the panel settings are kept by the browser.

## Where it came from

It grew out of a research tool that bred looks from a neuron map measured on one checkpoint. LoRA Casino keeps the
idea and drops everything tied to that checkpoint: the measured map (push sizes, groups, neighbours, tag names), the
fixed model, VAE and prompts. What it needs is worked out from whichever checkpoint you pick. Anima was added
afterwards, as the first model that is not SDXL.
