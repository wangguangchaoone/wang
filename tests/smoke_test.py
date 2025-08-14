import unittest
import torch

from yolomhaf import YOLOMHAF


class TestYOLOMHAF(unittest.TestCase):
    def test_forward_shapes(self):
        model = YOLOMHAF(num_classes=6)
        x = torch.randn(2, 3, 640, 640)
        preds = model(x)
        self.assertEqual(len(preds), 3)
        nc5 = 6 + 5
        expected = [
            (2, nc5, 80, 80),
            (2, nc5, 40, 40),
            (2, nc5, 20, 20),
        ]
        for p, exp in zip(preds, expected):
            self.assertEqual(tuple(p.shape), exp)


if __name__ == "__main__":
    unittest.main()