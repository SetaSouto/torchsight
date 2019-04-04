"""Test the IoU module."""
import unittest

import torch

from torchsight.metrics import iou as compute_iou


class TestIoU(unittest.TestCase):
    """Test the IoU module."""

    def __init__(self, *args, **kwargs):
        """Set the base boxes."""
        super(TestIoU, self).__init__(*args, **kwargs)

        self.boxes = torch.Tensor([[0.00, 0.00, 100., 100.],
                                   [0.00, 0.00, 200., 200.],
                                   [100., 100., 200., 200.],
                                   [200., 200., 400., 400.]])

    def test_iou(self):
        """Test that all the boxes with itself has an IoU of 1 and the correspondent with the
        rest."""
        iou = compute_iou(self.boxes, self.boxes)
        expected = torch.Tensor([[1.0000,  0.2500,  0.0000,  0.0000],
                                 [0.2500,  1.0000,  0.2500,  0.0000],
                                 [0.0000,  0.2500,  1.0000,  0.0000],
                                 [0.0000,  0.0000,  0.0000,  1.0000]])
        self.assertEqual(16, torch.eq(iou, expected).sum())


if __name__ == '__main__':
    unittest.main()
