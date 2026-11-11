"""The game without ComfyUI: run with  python -m unittest discover -s tests  (needs only numpy)."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_checkpoint  # noqa: E402
from casino import engine, weights  # noqa: E402


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    out = {k: np.frombuffer(raw[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], "<f4").reshape(v["shape"])
           for k, v in head.items() if k != "__metadata__"}
    return out, head.get("__metadata__", {})


class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.ckpt = fake_checkpoint.write(cls.dir / "fake_sdxl.safetensors")
        cls.St = weights._finish(weights.build(cls.ckpt))
        cls.kn = dict(engine.KNOBS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_is_read_as_sdxl(self):
        St = self.St
        self.assertEqual(len(St["names"]), 70)
        self.assertEqual(St["n"], sum(St["inner"]))
        self.assertTrue((St["absB"] > 0).all())
        self.assertTrue((St["layer"][St["knn"]] == St["layer"][:, None]).all(), "neighbours stay inside their layer")
        self.assertFalse((St["knn"] == np.arange(St["n"])[:, None]).any(), "a neuron is not its own neighbour")

    def test_other_models_are_refused_in_plain_words(self):
        small = fake_checkpoint.write(self.dir / "not_sdxl.safetensors", blocks=fake_checkpoint.SDXL[:3])
        with self.assertRaisesRegex(ValueError, "not an SDXL checkpoint"):
            weights.layers(small)
        empty = self.dir / "empty.safetensors"
        empty.write_bytes(struct.pack("<Q", 2) + b"{}")
        with self.assertRaisesRegex(ValueError, "no SDXL UNet"):
            weights.layers(empty)

    def test_f16_and_f32_checkpoints_give_the_same_numbers(self):
        b = weights.build(fake_checkpoint.write(self.dir / "f32.safetensors", dtype="F32"))
        self.assertTrue(np.allclose(self.St["absB"], b["absB"], rtol=0.05))

    def test_cards_come_in_four_sizes(self):
        rng = np.random.default_rng(1)
        for size, kind in ((1, "neuron"), (10, "cluster"), (100, "S"), (1000, "G")):
            p = engine.pick(self.St, size, rng=rng)
            self.assertEqual(p["kind"], kind)
            self.assertTrue(all(0 <= i < self.St["n"] for i in p["ids"]))
            self.assertEqual(len({int(self.St["layer"][i]) for i in p["ids"]}), 1, "a card sits in one layer")

    def test_every_move_gives_a_playable_hand(self):
        rng = np.random.default_rng(2)
        hand = [engine.ingredient(self.St, rng, 0.5, self.kn) for _ in range(3)]
        for move in ("tweak", "flip", "add", "swap", "drop", "remix", "leap"):
            for _ in range(20):
                child, caps = engine.mutate(self.St, hand, move, rng, 0.5, self.kn)
                self.assertTrue(child, move)
                self.assertEqual(len(engine.clean_mix(child, self.St["n"])), len(child), f"{move}: every card is valid")
                self.assertTrue(all(isinstance(c, str) for c in caps))

    def test_a_held_card_is_never_changed(self):
        rng = np.random.default_rng(3)
        hand = [engine.ingredient(self.St, rng, 0.5, self.kn) for _ in range(3)]
        hand[1]["held"] = True
        keep = (tuple(hand[1]["ids"]), hand[1]["s"])
        for wild in (0.0, 0.5, 1.0):
            for move in ("tweak", "flip", "add", "swap", "drop", "remix", "leap"):
                for _ in range(30):
                    child, _ = engine.mutate(self.St, hand, move, rng, wild, self.kn)
                    self.assertIn(keep, [(tuple(m["ids"]), m["s"]) for m in child], f"{move} at stakes {wild}")
        for fine in (0.0, 1.0):
            for child, _, _ in engine.polish(self.St, hand, 16, rng, fine):
                self.assertIn(keep, [(tuple(m["ids"]), m["s"]) for m in child], "refine")
        self.assertTrue(engine.clean_mix(hand, self.St["n"])[1].get("held"), "held survives the trip through the page")

    def test_a_hand_becomes_a_lora_comfyui_can_load(self):
        rng = np.random.default_rng(4)
        hand = [engine.pick(self.St, 100, rng=rng), engine.pick(self.St, 1, rng=rng)]
        out = self.dir / "look.safetensors"
        info = engine.write_lora(out, self.St, hand, self.ckpt, {"modelspec.title": "look"})
        tensors, meta = read_safetensors(out)
        self.assertEqual(meta["modelspec.title"], "look")
        self.assertEqual(len(tensors), len(info))
        for k, v in tensors.items():
            self.assertRegex(k, r"^lora_unet_.*_ff_net_2\.diff_b$")
            self.assertTrue(np.isfinite(v).all() and np.abs(v).sum() > 0)
        # the bias change is W @ shift: twice the hand is exactly twice the LoRA, and the opposite hand its negative
        double = [dict(m, s=2 * m["s"]) for m in hand]
        t2, _ = engine.lora_tensors(self.St, double, self.ckpt)
        for k, v in tensors.items():
            self.assertTrue(np.allclose(t2[k], 2 * v, rtol=1e-4, atol=1e-6))

    def test_the_same_hand_has_the_same_key(self):
        rng = np.random.default_rng(5)
        hand = [engine.pick(self.St, 10, rng=rng), engine.pick(self.St, 100, rng=rng)]
        self.assertEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, list(reversed(hand))))
        self.assertNotEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, hand[:1]))
        held = [dict(hand[0], held=True), hand[1]]
        self.assertEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, held), "holding a card does not change the picture")

    def test_a_card_is_described_for_the_page(self):
        rng = np.random.default_rng(6)
        p = engine.pick(self.St, 100, rng=rng, sign=1)
        d = engine.describe(self.St, p)
        self.assertEqual(d["n"], len(p["ids"]))
        self.assertTrue(d["up"])
        self.assertIn(d["where"], ("layout", "style", "fine detail"))
        self.assertEqual(d["cid"], engine.describe(self.St, dict(p, s=-p["s"]))["cid"], "a card turned over is the same card")

    def test_png_text_keeps_the_picture_readable(self):
        import zlib
        def chunk(t, b):
            return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b"")
        out = engine.png_with_text(png, {"prompt": '{"a": 1}'})
        self.assertEqual(out[:8 + 25], png[:8 + 25])
        self.assertIn(b'tEXtprompt\x00{"a": 1}', out)
        self.assertTrue(out.endswith(png[8 + 25:]))


if __name__ == "__main__":
    unittest.main()
