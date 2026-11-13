"""The game without ComfyUI: run with  python -m unittest discover -s tests  (needs only numpy).

The same game is played on every family of model, so the tests of the game (Game) run once per family, on a tiny
stand-in file for each; what is particular to a family is tested in its own class."""
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
from casino import engine, families, weights  # noqa: E402


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    out = {k: np.frombuffer(raw[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], "<f4").reshape(v["shape"])
           for k, v in head.items() if k != "__metadata__"}
    return out, head.get("__metadata__", {})


class Game:
    """Tests of the game itself; a subclass sets `make` (writes the stand-in file) and `suits`."""
    make = suits = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.ckpt = cls.make(cls.dir / "fake.safetensors")
        cls.St = weights._finish(weights.build(cls.ckpt))
        cls.kn = dict(engine.KNOBS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_numbers_cover_every_neuron(self):
        St = self.St
        self.assertEqual(St["n"], sum(St["inner"]))
        self.assertTrue((St["absB"] > 0).all())
        self.assertTrue((St["layer"][St["knn"]] == St["layer"][:, None]).all(), "neighbours stay inside their layer")
        self.assertFalse((St["knn"] == np.arange(St["n"])[:, None]).any(), "a neuron is not its own neighbour")
        self.assertEqual(set(St["zone"]), set(self.suits))
        for k in ("wkey", "lora", "push", "zone", "detail", "s0"):
            self.assertEqual(len(St[k]), len(St["names"]), k)

    def test_f16_and_f32_files_give_the_same_numbers(self):
        b = weights.build(self.make(self.dir / "f32.safetensors", dtype="F32"))
        self.assertTrue(np.allclose(self.St["absB"], b["absB"], rtol=0.05))

    def test_the_numbers_survive_being_kept(self):
        weights.CACHE = self.dir / "cache"
        key, f = weights.cache_file(self.ckpt)
        weights._state[key] = {"p": 0.0, "error": None, "stats": None}
        weights._work(self.ckpt, key, f)
        self.assertIsNone(weights._state[key]["error"])
        back = weights._load(f)
        self.assertEqual(back["family"], self.St["family"])
        self.assertEqual(back["lora"], self.St["lora"])
        self.assertTrue(np.array_equal(back["g2"], self.St["g2"]))

    def test_cards_come_in_four_sizes(self):
        rng = np.random.default_rng(1)
        for size, kind in ((1, "neuron"), (10, "cluster"), (100, "S"), (1000, "G")):
            for _ in range(20):
                p = engine.pick(self.St, size, rng=rng)
                self.assertEqual(p["kind"], kind)
                self.assertTrue(all(0 <= i < self.St["n"] for i in p["ids"]))
                self.assertEqual(len({int(self.St["layer"][i]) for i in p["ids"]}), 1, "a card sits in one layer")
                self.assertGreater(abs(p["s"]), 0)

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

    def test_any_hand_becomes_a_lora_and_it_is_linear(self):
        rng = np.random.default_rng(4)
        for _ in range(12):
            hand = [engine.ingredient(self.St, rng, 0.5, self.kn) for _ in range(4)]
            out = self.dir / "look.safetensors"
            info = engine.write_lora(out, self.St, hand, self.ckpt, {"modelspec.title": "look"})
            tensors, meta = read_safetensors(out)
            self.assertEqual(meta["modelspec.title"], "look")
            self.assertTrue(info)
            for k, v in tensors.items():
                self.assertRegex(k, r"^lora_unet_\w+\.(diff_b|lora_up\.weight|lora_down\.weight|alpha)$")
                self.assertTrue(np.isfinite(v).all() and np.abs(v).sum() > 0, k)
            # twice the hand is twice the change, the opposite hand its negative
            t2, _ = engine.lora_tensors(self.St, [dict(m, s=-2 * m["s"]) for m in hand], self.ckpt)
            self.assertEqual(set(t2), set(tensors))
            for k, v in tensors.items():
                if k.endswith(".diff_b") or k.endswith(".lora_down.weight"):
                    self.assertTrue(np.allclose(t2[k], -2 * v, rtol=1e-4, atol=1e-6), k)
                else:   # the direction (lora_up) and alpha are the card's own, whatever its strength
                    self.assertTrue(np.allclose(t2[k], v, rtol=1e-4, atol=1e-6), k)

    def test_the_same_hand_has_the_same_key(self):
        rng = np.random.default_rng(5)
        for _ in range(12):
            hand = [engine.ingredient(self.St, rng, 0.5, self.kn), engine.ingredient(self.St, rng, 0.5, self.kn)]
            if set(hand[0]["ids"]) == set(hand[1]["ids"]):
                continue
            self.assertEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, list(reversed(hand))))
            self.assertNotEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, hand[:1]))
            self.assertNotEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, [hand[0], dict(hand[1], s=hand[1]["s"] * 1.5)]))
            held = [dict(hand[0], held=True), hand[1]]
            self.assertEqual(engine.mix_key(self.St, hand), engine.mix_key(self.St, held), "holding a card does not change the picture")

    def test_a_card_is_described_for_the_page(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            p = engine.ingredient(self.St, rng, 0.5, self.kn)
            d = engine.describe(self.St, p)
            self.assertEqual(d["n"], len(p["ids"]))
            self.assertEqual(d["up"], p["s"] > 0)
            self.assertIn(d["where"], self.suits)
            self.assertIn(d["power"], ("gentle", "medium", "strong", "very strong"))
            self.assertEqual(d["cid"], engine.describe(self.St, dict(p, s=-p["s"]))["cid"], "a card turned over is the same card")


class Sdxl(Game, unittest.TestCase):
    make = staticmethod(fake_checkpoint.write)
    suits = ("layout", "style", "fine detail")

    def test_the_file_is_read_as_sdxl(self):
        self.assertEqual(self.St["family"], "sdxl")
        self.assertEqual(len(self.St["names"]), 70)
        self.assertEqual(set(self.St["push"]), {"bias"})
        self.assertTrue((self.St["unit"] == 0).all(), "SDXL strengths are multiples of each neuron's typical activation")

    def test_a_hand_is_a_bias_change(self):
        rng = np.random.default_rng(7)
        hand = [engine.pick(self.St, 100, rng=rng), engine.pick(self.St, 1, rng=rng)]
        tensors, info = engine.lora_tensors(self.St, hand, self.ckpt)
        self.assertEqual(len(tensors), len(info))
        for k in tensors:
            self.assertRegex(k, r"^lora_unet_.*_ff_net_2\.diff_b$")

    def test_numbers_kept_before_anima_still_load(self):
        St = self.St
        f = self.dir / "old.npz"
        np.savez(f, names=np.array(json.dumps(St["names"])), inner=np.array(St["inner"]), off=St["off"], layer=St["layer"],
                 idx=St["idx"], absB=St["absB"], knn=St["knn"], g1=St["g1"], g2=St["g2"])
        back = weights._load(f)
        self.assertEqual(back["family"], "sdxl")
        self.assertEqual(back["wkey"], St["wkey"])

    def test_png_text_keeps_the_picture_readable(self):
        import zlib
        def chunk(t, b):
            return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b"")
        out = engine.png_with_text(png, {"prompt": '{"a": 1}'})
        self.assertEqual(out[:8 + 25], png[:8 + 25])
        self.assertIn(b'tEXtprompt\x00{"a": 1}', out)
        self.assertTrue(out.endswith(png[8 + 25:]))


class Anima(Game, unittest.TestCase):
    make = staticmethod(fake_checkpoint.write_anima)
    suits = ("prompt", "layout", "subject", "rendering")

    def test_the_file_is_read_as_anima(self):
        St = self.St
        self.assertEqual(St["family"], "anima")
        self.assertEqual(len(St["names"]), 12)
        self.assertEqual(St["zone"], ["prompt"] * 3 + ["layout"] * 3 + ["subject"] * 3 + ["rendering"] * 3, "the prompt is read first")
        self.assertEqual(St["push"], ["bias"] * 3 + ["group"] * 9, "only the text adapter has biases")
        self.assertTrue((St["unit"] > 0).all(), "every layer's whole output has a size: strengths count against it")
        self.assertTrue((St["var"][St["off"][3]:] > 0).all() and np.isfinite(St["act"]).all())
        self.assertTrue(np.abs(St["gate"][3:, :16]).sum() > 0)

    def test_a_file_without_the_timestep_weights_still_plays(self):
        St = weights._finish(weights.build(fake_checkpoint.write_anima(self.dir / "noclock.safetensors", clock=False)))
        self.assertTrue((St["unit"] > 0).all())
        self.assertTrue((St["gate"][3:, :16] == 1).all(), "as if the blocks did not modulate")
        rng = np.random.default_rng(8)
        t, _ = engine.lora_tensors(St, [engine.ingredient(St, rng, 0.5, self.kn) for _ in range(4)], self.dir / "noclock.safetensors")
        self.assertTrue(all(np.isfinite(v).all() for v in t.values()))

    def test_a_file_that_names_its_tensors_another_way_is_anima_too(self):
        # "net." is how the Anima base release names them; "" is a file saved with bare names
        for prefix in ("net.", "", "some.other.wrapper."):
            f = fake_checkpoint.write_anima(self.dir / f"named_{len(prefix)}.safetensors", prefix=prefix)
            m = weights.model(f)
            self.assertEqual((m["family"], m["prefix"], len(m["layers"])), ("anima", prefix, 12), prefix)
            St = weights._finish(weights.build(f))
            self.assertTrue(all(k.startswith(prefix + ("blocks." if i >= 3 else "llm_adapter.")) for i, k in enumerate(St["wkey"])), prefix)
            self.assertEqual(St["lora"], self.St["lora"], "the LoRA keys are the same either way: ComfyUI names the loaded model's layers itself")
            self.assertTrue(np.allclose(St["unit"], self.St["unit"], rtol=1e-4), "and so are the numbers, timestep weights included")

    def test_one_neuron_in_a_block_is_made_exactly_louder(self):
        St = self.St
        g = int(St["off"][5]) + 7                       # a neuron of the third block
        W = weights.tensor(self.ckpt, St["wkey"][5])
        t, _ = engine.lora_tensors(St, [{"ids": [g], "s": 2.5, "label": "", "kind": "neuron", "center": g}], self.ckpt)
        base = St["lora"][5]
        self.assertEqual(set(t), {f"{base}.lora_up.weight", f"{base}.lora_down.weight", f"{base}.alpha"})
        change = t[f"{base}.lora_up.weight"] @ t[f"{base}.lora_down.weight"] * float(t[f"{base}.alpha"]) / 1   # rank 1: alpha / rank = 1
        a = change[:, 7] @ W[:, 7] / (W[:, 7] @ W[:, 7])
        want = np.zeros_like(W)
        want[:, 7] = a * W[:, 7]                        # only this neuron changes: it writes 1 + a times as much
        self.assertGreater(a, 0)
        self.assertTrue(np.allclose(change, want, rtol=1e-4, atol=1e-6))

    def test_the_same_strength_is_the_same_size_of_push_on_any_card(self):
        St = self.St
        rng = np.random.default_rng(9)
        for size in (1, 10, 100, 1000):
            for _ in range(6):
                card = engine.pick(St, size, int(rng.integers(St["off"][3], St["n"])), rng)
                li = int(St["layer"][card["center"]])
                idx = np.sort(St["idx"][np.array(card["ids"])].astype(np.int64))
                t, _ = engine.lora_tensors(St, [card], self.ckpt)
                up, down = t[f"{St['lora'][li]}.lora_up.weight"][:, 0], t[f"{St['lora'][li]}.lora_down.weight"][0]
                e, var = St["act"][St["off"][li] + idx], St["var"][St["off"][li] + idx]
                push = np.linalg.norm(St["gate"][li][:16] * up) * np.sqrt((down[idx] @ e) ** 2 + (down[idx] ** 2) @ var)
                self.assertAlmostEqual(push / St["unit"][li], abs(card["s"]), delta=0.02 * abs(card["s"]), msg=f"{size} neurons")

    def test_cards_in_one_block_share_its_lora_and_stay_apart(self):
        St = self.St
        a = int(St["off"][8])
        hand = [{"ids": list(range(a, a + 10)), "s": 3.0, "label": "", "kind": "cluster", "center": a},
                {"ids": list(range(a + 40, a + 50)), "s": -2.0, "label": "", "kind": "cluster", "center": a + 40}]
        t, info = engine.lora_tensors(St, hand, self.ckpt)
        base = St["lora"][8]
        up, down = t[f"{base}.lora_up.weight"], t[f"{base}.lora_down.weight"]
        self.assertEqual((up.shape, down.shape), ((16, 2), (2, 128)), "one rank per card")
        self.assertEqual(float(t[f"{base}.alpha"]), 2.0)
        self.assertTrue((down[0, 10:] == 0).all() and (down[1, :40] == 0).all() and (down[1, 50:] == 0).all(), "a card touches only its neurons")
        W = weights.tensor(self.ckpt, St["wkey"][8])
        self.assertTrue(np.allclose(up[:, 0], W[:, :10].mean(1), rtol=1e-4, atol=1e-6), "the push goes along what the card's neurons write on average")
        self.assertEqual(info[0]["cards"], 2)

    def test_a_prompt_card_is_a_bias_change(self):
        St = self.St
        g = int(St["off"][1]) + 3
        t, _ = engine.lora_tensors(St, [{"ids": [g, g + 1], "s": 4.0, "label": "", "kind": "cluster", "center": g}], self.ckpt)
        self.assertEqual(list(t), ["lora_unet_llm_adapter_blocks_1_mlp_2.diff_b"])
        W = weights.tensor(self.ckpt, St["wkey"][1])
        v = W[:, 3] * St["absB"][g] + W[:, 4] * St["absB"][g + 1]      # each neuron moved in proportion to its usual size
        db = t["lora_unet_llm_adapter_blocks_1_mlp_2.diff_b"]
        self.assertTrue(np.allclose(db / np.linalg.norm(db), v / np.linalg.norm(v), rtol=1e-3, atol=1e-5))
        self.assertAlmostEqual(float(np.linalg.norm(db)), 4.0 * float(St["unit"][1]), delta=0.01 * 4.0 * float(St["unit"][1]))


class OtherFiles(unittest.TestCase):
    def test_other_models_are_refused_in_plain_words(self):
        with tempfile.TemporaryDirectory() as d:
            small = fake_checkpoint.write(Path(d) / "not_sdxl.safetensors", blocks=fake_checkpoint.SDXL[:3])
            with self.assertRaisesRegex(ValueError, "not an SDXL checkpoint"):
                weights.layers(small)
            empty = Path(d) / "empty.safetensors"
            empty.write_bytes(struct.pack("<Q", 2) + b"{}")
            with self.assertRaisesRegex(ValueError, "not a model LoRA Casino can play on"):
                weights.layers(empty)

    def test_every_family_says_how_it_is_drawn(self):
        for name, F in families.FAMILY.items():
            self.assertTrue({"negative", "cfg", "sampler", "scheduler", "steps"} <= set(F["draw"]), name)
            self.assertIn("title", F)
            self.assertTrue(families.lora_meta(name))


if __name__ == "__main__":
    unittest.main()
