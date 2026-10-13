import unittest
import numpy as np

from neurreps.geometry import linear_cka, participation_ratio, counterfactual_direction_geometry
from neurreps.models import build_model
from neurreps.data import connected_match_groups
from neurreps.counterfactual import generate_matched_counterfactual_pairs


class TestGeometry(unittest.TestCase):
    def test_cka_identity(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 12))
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=8)

    def test_participation_ratio(self):
        x = np.eye(8)
        self.assertGreater(participation_ratio(x), 1.0)

    def test_models_accept_shapes(self):
        import torch
        for rep, shape in [("SC", (4, 1, 64)), ("4C", (4, 4, 16)), ("32C", (4, 32, 16))]:
            model = build_model(rep, width=8, latent_dim=6, depth=1)
            logits, z = model(torch.zeros(shape), return_latent=True)
            self.assertEqual(tuple(logits.shape), (4,))
            self.assertEqual(tuple(z.shape), (4, 6))

    def test_counterfactual(self):
        b = np.zeros((10, 4))
        m = np.ones((10, 4))
        r = counterfactual_direction_geometry(b, m)
        self.assertAlmostEqual(r["direction_concentration"], 1.0, places=7)

    def test_connected_match_groups_preserve_pairs_and_duplicates(self):
        hashes = np.array(["same", "b", "same", "d"])
        selected = np.array([0, 1, 2, 3])
        pairs = np.array([[0, 1], [2, 3]])
        groups = connected_match_groups(hashes, selected, pairs)
        self.assertEqual(len(set(groups.tolist())), 1)

    def test_counterfactual_allowed_hash_filter(self):
        common = {
            "records_s1": [
                {"sample_hash": "keep", "source_index": 1, "chain_blocks": ["A", "B", "C"]},
                {"sample_hash": "drop", "source_index": 2, "chain_blocks": ["A", "B", "C"]},
            ],
            "bb2instr": {key: [1] for key in ["A", "B", "C", "D", "E", "F", "G"]},
            "SUCC_MAP": {
                "A": {"B"}, "B": {"C", "D"}, "C": set(), "D": set(),
                "E": set(), "F": {"B"}, "G": set(),
            },
        }
        pairs = generate_matched_counterfactual_pairs(
            common, max_pairs=2, seed=0, weak_blocks={"B", "G"},
            length_tolerance=0, allowed_sample_hashes={"keep"},
        )
        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(all(p["benign_sample_hash"] == "keep" for p in pairs))


if __name__ == "__main__":
    unittest.main()
