"""Tiny stand-ins for real model files: the same feed-forward layers under the same names, with small random weights
(a few hundred kB instead of several GB). Enough for everything LoRA Casino reads from a model, so the engine can be
tested on any machine, without ComfyUI, a GPU or a model download.

  write()        an SDXL checkpoint: its 70 layers
  write_anima()  an Anima model: a few bias-free blocks and the text adapter's blocks (which have biases)
"""
import json
import struct

import numpy as np

# SDXL's transformers: (where, blocks inside). Widths are shrunk; the 2 : 10 block pattern is SDXL's own.
SDXL = [("input_blocks.4.1", 2), ("input_blocks.5.1", 2), ("input_blocks.7.1", 10), ("input_blocks.8.1", 10), ("middle_block.1", 10),
        ("output_blocks.0.1", 10), ("output_blocks.1.1", 10), ("output_blocks.2.1", 10),
        ("output_blocks.3.1", 2), ("output_blocks.4.1", 2), ("output_blocks.5.1", 2)]


def _save(path, tensors, dtype):
    np_type = {"F16": np.float16, "F32": np.float32}[dtype]
    head, blobs, at = {}, [], 0
    for k, v in tensors.items():
        b = np.asarray(v, np_type).tobytes()
        head[k] = {"dtype": dtype, "shape": list(np.shape(v)), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    hj = json.dumps(head).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))
    return path


def write(path, seed=0, dtype="F16", blocks=SDXL):
    rng = np.random.default_rng(seed)
    tensors = {}
    for where, n in blocks:
        dim = 8 if n == 2 else 16
        inner = dim * 4
        for b in range(n):
            p = f"model.diffusion_model.{where}.transformer_blocks.{b}"
            tensors[p + ".norm3.weight"] = 1 + 0.1 * rng.standard_normal(dim)
            tensors[p + ".norm3.bias"] = 0.1 * rng.standard_normal(dim)
            tensors[p + ".ff.net.0.proj.weight"] = 0.3 * rng.standard_normal((inner * 2, dim))
            tensors[p + ".ff.net.0.proj.bias"] = 0.1 * rng.standard_normal(inner * 2)
            tensors[p + ".ff.net.2.weight"] = 0.3 * rng.standard_normal((dim, inner))
            tensors[p + ".ff.net.2.bias"] = 0.1 * rng.standard_normal(dim)
    return _save(path, tensors, dtype)


def write_anima(path, seed=0, dtype="F16", blocks=9, text_blocks=3, prefix="model.diffusion_model.", clock=True):
    """Anima as ComfyUI stores it: blocks.N.mlp.layer1 / layer2 without biases, llm_adapter.blocks.N.mlp.0 / .2 with,
    and (clock) the weights that tell each block the timestep. prefix "" = a file saved with bare names, which also
    happens."""
    rng = np.random.default_rng(seed)
    tensors = {}
    if clock:
        tensors[prefix + "t_embedder.1.linear_1.weight"] = 0.3 * rng.standard_normal((16, 16))
        tensors[prefix + "t_embedder.1.linear_2.weight"] = 0.1 * rng.standard_normal((48, 16))
        tensors[prefix + "t_embedding_norm.weight"] = 1 + 0.1 * rng.standard_normal(16)
        for b in range(blocks):
            tensors[f"{prefix}blocks.{b}.adaln_modulation_mlp.1.weight"] = 0.3 * rng.standard_normal((4, 16))
            tensors[f"{prefix}blocks.{b}.adaln_modulation_mlp.2.weight"] = 0.3 * rng.standard_normal((48, 4))
    for b in range(text_blocks):
        p, dim, inner = f"{prefix}llm_adapter.blocks.{b}", 8, 48
        tensors[p + ".norm_mlp.weight"] = 1 + 0.1 * rng.standard_normal(dim)
        tensors[p + ".mlp.0.weight"] = 0.3 * rng.standard_normal((inner, dim))
        tensors[p + ".mlp.0.bias"] = 0.1 * rng.standard_normal(inner)
        tensors[p + ".mlp.2.weight"] = 0.3 * rng.standard_normal((dim, inner))
        tensors[p + ".mlp.2.bias"] = 0.1 * rng.standard_normal(dim)
    for b in range(blocks):
        p, dim, inner = f"{prefix}blocks.{b}", 16, 128
        tensors[p + ".mlp.layer1.weight"] = 0.3 * rng.standard_normal((inner, dim))
        tensors[p + ".mlp.layer2.weight"] = 0.3 * rng.standard_normal((dim, inner))
    return _save(path, tensors, dtype)
