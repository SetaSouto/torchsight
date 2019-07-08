"""Module with test for the retrieval metrics."""
import unittest

import torch

from torchsight.metrics.retrieval import AveragePrecision


class TestAveragePrecision(unittest.TestCase):
    """Test the average precision metric."""

    def test_correctness(self):
        """Test that the metric give correct results."""
        metric = AveragePrecision()
        results = torch.Tensor([[1, 1, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]])
        avgs = metric(results)

        self.assertEqual(float(avgs[0]), 1)
        self.assertEqual(float(avgs[1]), 0.25)
        self.assertEqual(float(avgs[2]), 0.75)


if __name__ == '__main__':
    unittest.main()
