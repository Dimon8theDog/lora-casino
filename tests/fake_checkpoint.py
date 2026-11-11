"""A tiny stand-in for an SDXL checkpoint: the same 70 feed-forward layers under the same names, with small random
weights (a few hundred kB instead of 6 GB). Enough for everything LoRA Casino reads from a checkpoint, so the engine
can be tested on any machine, without ComfyUI, a GPU or a model download."""
import json
import struct

import numpy as np

# SDXL's transformers: (where, blocks inside). Widths are shrunk; the 2 : 10 block pattern is SDXL's own.
SDXL = [("input_blocks.4.1", 2), ("input_blocks.5.1", 2), ("input_blocks.7.1", 10), ("input_blocks.8.1", 10), ("middle_block.1", 10),
        ("output_blocks.0.1", 10), ("output_blocks.1.1", 10), ("output_blocks.2.1", 10),
        ("output_blocks.3.1", 2), ("output_blocks.4.1", 2), ("output_blocks.5.1", 2)]


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
