import unittest

import numpy as np
import pandas as pd
import torch

from neurreps.ablations import (
    append_explicit_mask_channel,
    randomize_padding_positions,
    replace_padding_with_training_content,
)
from neurreps.data import exact_length_matched_pairs, no_padding_indices
from neurreps.models import build_model


class TestPaddingAblations(unittest.TestCase):
    def test_explicit_mask_channel(self):
        X = np.zeros((2, 4, 5), dtype=np.uint8)
        mask = np.array([[[1, 1, 1, 0, 0]], [[1, 1, 0, 0, 0]]], dtype=np.uint8)
        out = append_explicit_mask_channel(X, mask, "4C")
        self.assertEqual(out.shape, (2, 5, 5))
        np.testing.assert_array_equal(out[:, -1:], mask * 255)

    def test_randomized_padding_preserves_order_and_count(self):
        X = np.array([[[10, 11, 12, 0, 0, 0]]], dtype=np.uint8)
        mask = np.array([[[1, 1, 1, 0, 0, 0]]], dtype=np.uint8)
        out, out_mask = randomize_padding_positions(X, mask, seed=7)
        valid = out[0, 0, out_mask[0, 0] == 1]
        padded = out[0, 0, out_mask[0, 0] == 0]
        np.testing.assert_array_equal(valid, np.array([10, 11, 12], dtype=np.uint8))
        np.testing.assert_array_equal(padded, np.array([0, 0, 0], dtype=np.uint8))
        self.assertEqual(int(out_mask.sum()), 3)

    def test_zero_cue_replacement_uses_training_content(self):
        X = np.array(
            [
                [[3, 4, 0, 0]],
                [[7, 8, 0, 0]],
                [[9, 10, 0, 0]],
            ],
            dtype=np.uint8,
        )
        mask = np.array([[[1, 1, 0, 0]]] * 3, dtype=np.uint8)
        out = replace_padding_with_training_content(X, mask, np.array([0, 1]), seed=1)
        self.assertTrue(np.all(np.isin(out[:, :, 2:], [3, 4, 7, 8])))
        np.testing.assert_array_equal(out[:, :, :2], X[:, :, :2])

    def test_exact_length_matching(self):
        meta = pd.DataFrame(
            {
                "sc_real_length": [8, 8, 9, 8, 8, 10],
                "slot_real_length": [2, 2, 3, 2, 2, 4],
            }
        )
        y = np.array([0, 0, 0, 1, 1, 1])
        pairs = exact_length_matched_pairs(meta, y, seed=0)
        self.assertEqual(pairs.shape, (2, 2))
        for a, b in pairs:
            self.assertEqual(y[a], 0)
            self.assertEqual(y[b], 1)
            self.assertEqual(tuple(meta.iloc[a]), tuple(meta.iloc[b]))

    def test_no_padding_intersection(self):
        masks = {
            "SC": np.array([[[1, 1]], [[1, 0]], [[1, 1]]]),
            "4C": np.array([[[1, 1]], [[1, 1]], [[1, 0]]]),
            "32C": np.array([[[1, 1]], [[1, 1]], [[1, 0]]]),
        }
        idx = no_padding_indices(masks)
        np.testing.assert_array_equal(idx, np.array([0]))

    def test_mask_aware_encoder_ignores_padded_values(self):
        torch.manual_seed(0)
        model = build_model("4C", width=8, latent_dim=5, depth=1, mask_aware=True)
        model.eval()
        mask = torch.tensor([[[1, 1, 0, 0]]], dtype=torch.float32)
        x1 = torch.zeros((1, 4, 4), dtype=torch.uint8)
        x2 = x1.clone()
        x1[:, :, :2] = 10
        x2[:, :, :2] = 10
        x2[:, :, 2:] = 255
        with torch.no_grad():
            z1 = model.encode(x1, mask=mask)
            z2 = model.encode(x2, mask=mask)
        self.assertTrue(torch.allclose(z1, z2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
