"""What LoRA Casino knows about each kind of model it can play on. Everything else in the game is the same for all.

A family says:
  how to recognise its file        read(): from the names and shapes in the .safetensors header
  where its neurons are            one feed-forward layer per transformer block; a neuron is one hidden unit of it
  what a push is there             "bias"   the layer has a bias: moving its neurons by a fixed amount is the same as
                                            changing that bias (W d), saved as a 'diff_b' LoRA key. Exact.
                                   "group"  the layer has no bias. The card's neurons get one shared push along the
                                            direction they write in on average: a rank-1 change of the layer's weight,
                                            saved as ordinary lora_up / lora_down keys. It grows with how strongly the
                                            neurons fire.
  which suit a layer's cards are   by where the layer sits
  how strong a card starts         per card size (what the number counts is in engine.py)
  how to draw it                   the loader nodes and the settings its makers recommend

SDXL (and Pony, Illustrious, NoobAI ...): 70 layers in a UNet, all with biases.
Anima (the Cosmos-Predict2 transformer with a small text adapter): 28 blocks without any bias, plus the 6 blocks of the
text adapter, which do have one. The adapter's cards change how the prompt is read; the blocks' cards change the
picture: its layout early, what is in it in the middle, how it is rendered late. That order was seen on one checkpoint
(anima-aesthetic-v1.1), one block from each third and two adapter blocks; the borders between the thirds are a guess.
"""
import re

SDXL_PREFIX = "model.diffusion_model."
SDXL_FF = re.compile(r"^model\.diffusion_model\.((input_blocks|middle_block|output_blocks)\.(?:(\d+)\.)?(\d+)"
                     r"\.transformer_blocks\.(\d+))\.ff\.net\.2\.weight$")
SDXL_LAYERS = 70
# SDXL's own structure (the same in every SDXL checkpoint): the middle of the decoder sets the look, its last,
# high-resolution blocks paint fine detail, everything before lays the picture out
SDXL_STYLE = ("output_blocks.0.1", "output_blocks.1.1", "output_blocks.2.1")
SDXL_DETAIL = ("output_blocks.3.1", "output_blocks.4.1", "output_blocks.5.1")

ANIMA_BLOCK = re.compile(r"^(model\.diffusion_model\.)?(blocks\.(\d+))\.mlp\.layer2\.weight$")
ANIMA_TEXT = re.compile(r"^(model\.diffusion_model\.)?(llm_adapter\.blocks\.(\d+))\.mlp\.2\.weight$")

# How strong a fresh card of 1 / 10 / 100 / 1000 neurons starts ("Card strength" in the page scales all of them; a
# deal at the default settings plays cards at about a quarter to four fifths of this).
# SDXL: multiples of each neuron's typical activation, so a card of fewer neurons needs more.
# Anima: the whole push against its layer's whole output (engine.py), so one number does for every size.
STRENGTH = {
    "sdxl": {1: 300.0, 10: 100.0, 100: 60.0, 1000: 6.0},
    "anima": {1: 2.8, 10: 2.8, 100: 2.8, 1000: 2.8},
    "anima-text": {1: 1.0, 10: 1.0, 100: 1.0, 1000: 1.0},    # the prompt is read in fewer, narrower layers: less goes further
}
# Anima's suits differ in how soon a picture breaks. Measured on anima-aesthetic-v1.1 with 16 cards per suit on two
# prompts (a picture counted as broken when it turned to noise, stripes or a flat wash):
#   one card at 1.2   Layout 0 of 32 broken, Subject 4, Rendering 3
#   one card at 2.4   Layout 2 of 32 broken, Subject 11, Rendering 9
# So Subject and Rendering start lower: a deal at the default settings plays Layout cards at about 1.3 (up to 2.2),
# the other two at about 1.0 (up to 1.7). Layout does not start higher, although its cards move a picture less at the
# same strength: in real deals, hands with a Layout card near 3 broke.
STURDY = {"layout": 1.0, "subject": 0.75, "rendering": 0.75}

FAMILY = {
    "sdxl": {
        "title": "SDXL",
        "draw": {"negative": "low quality, worst quality, blurry", "cfg": 5.0, "sampler": "dpmpp_2m", "scheduler": "karras", "steps": 12},
        "seconds": 1.0,          # how long a picture takes next to SDXL, for the page's time guess
    },
    "anima": {
        "title": "Anima",
        "draw": {"negative": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                 "cfg": 4.0, "sampler": "er_sde", "scheduler": "simple", "steps": 16},
        "seconds": 1.1,          # per step and pixel it is about SDXL's speed (measured on one 3080 Ti)
        # The page's sliders start here for Anima. Where new cards come from: mostly Layout and Subject, some Prompt,
        # few Rendering (late blocks mostly blur or roughen a small picture). Cards in a new hand: one or two, because
        # one Anima card already changes a picture visibly and each further card is another chance to break it.
        "knobs": {"depth": 0.35, "spread": 0.4, "count": 0.15},
        "clip_type": "stable_diffusion",                 # what ComfyUI's CLIPLoader is told for its text encoder
        "likes": {"te": "qwen_3_06b", "vae": "qwen_image_vae"},   # the files its makers ship, to pick by default
    },
}


def read(header, filename):
    """Which family a .safetensors header belongs to -> {"family", "prefix", "layers": [(name, neurons, width)]}.
    Layers are in the order the model runs them. ValueError if it is neither."""
    sdxl, blocks, text, prefix = [], [], [], ""
    for k, t in header.items():
        m = SDXL_FF.match(k)
        if m:
            part = {"input_blocks": 0, "middle_block": 1, "output_blocks": 2}[m.group(2)]
            sdxl.append(((part, int(m.group(3) or 0), int(m.group(4)), int(m.group(5))), m.group(1), t["shape"][1], t["shape"][0]))
            continue
        m = ANIMA_BLOCK.match(k) or ANIMA_TEXT.match(k)
        if m:
            prefix = m.group(1) or ""
            (blocks if m.group(2).startswith("blocks") else text).append((int(m.group(3)), m.group(2), t["shape"][1], t["shape"][0]))
    if blocks and text:
        found = sorted(text) + sorted(blocks)          # the prompt is read first, then the picture is made
        return {"family": "anima", "prefix": prefix, "layers": [(n, int(inner), int(dim)) for _, n, inner, dim in found]}
    if sdxl:
        if len(sdxl) != SDXL_LAYERS:
            raise ValueError(f"{filename} is not an SDXL checkpoint: it has {len(sdxl)} transformer layers, SDXL has {SDXL_LAYERS}")
        return {"family": "sdxl", "prefix": SDXL_PREFIX, "layers": [(n, int(inner), int(dim)) for _, n, inner, dim in sorted(sdxl)]}
    raise ValueError(f"{filename} is not a model LoRA Casino can play on: it knows SDXL checkpoints and Anima "
                     "(this looks like another kind of model, or a LoRA / VAE file)")


def per_layer(family, prefix, names):
    """For each layer: the key of its write weight, its LoRA key, the kind of push, its suit, whether small 'detail'
    touches come from it, and how strong its cards start."""
    out = {k: [] for k in ("wkey", "lora", "push", "zone", "detail", "s0")}

    def add(wkey, lora, push, zone, detail, s0):
        for k, v in zip(out, (wkey, lora, push, zone, detail, s0)):
            out[k].append(v)

    if family == "sdxl":
        for n in names:
            block = n.split(".transformer")[0]
            zone = "style" if block in SDXL_STYLE else "fine detail" if block.startswith("output_blocks") else "layout"
            add(f"{prefix}{n}.ff.net.2.weight", "lora_unet_" + f"{n}.ff.net.2".replace(".", "_"), "bias", zone,
                block in SDXL_DETAIL, STRENGTH["sdxl"])
        return out
    main = [n for n in names if n.startswith("blocks.")]
    for n in names:
        if n.startswith("llm_adapter."):
            add(f"{prefix}{n}.mlp.2.weight", "lora_unet_" + f"{n}.mlp.2".replace(".", "_"), "bias", "prompt", False, STRENGTH["anima-text"])
            continue
        at = main.index(n) / max(1, len(main) - 1)
        zone = "layout" if at < 1 / 3 else "subject" if at < 2 / 3 else "rendering"
        add(f"{prefix}{n}.mlp.layer2.weight", "lora_unet_" + f"{n}.mlp.layer2".replace(".", "_"), "group", zone,
            at > 0.82, {k: round(v * STURDY[zone], 3) for k, v in STRENGTH["anima"].items()})
    return out


def lora_meta(family):
    if family == "sdxl":
        return {"ss_network_module": "loracasino.bias_diff", "ss_base_model_version": "sdxl_base_v1-0",
                "modelspec.architecture": "stable-diffusion-xl-v1-base/lora"}
    return {"ss_network_module": "loracasino.push", "loracasino_family": family}
