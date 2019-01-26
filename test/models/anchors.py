"""Test for the Anchors model."""
import unittest

import torch

from torchsight.models import Anchors


class TestAnchors(unittest.TestCase):
    """Test that the anchors model works well."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super(TestAnchors, self).__init__(*args, **kwargs)

        self.sizes = [32, 64, 128, 256, 512]
        self.scales = [2 ** 0, 2 ** (1/3), 2 ** (2/3)]
        self.ratios = [0.5, 1, 2]
        self.strides = [8, 16, 32, 64, 128]
        self.anchors = Anchors(self.sizes, self.scales, self.ratios, self.strides)

    def test_base_anchors(self):
        """Test that constructs the base anchors well with all the scales and ratios."""
        base_anchors = self.anchors.base_anchors
        # Test the shape
        self.assertEqual(len(self.sizes), base_anchors.shape[0])
        self.assertEqual(len(self.ratios) * len(self.scales), base_anchors.shape[1])
        self.assertEqual(4, base_anchors.shape[2])
        # Test that each anchor has the correct area and aspect ratio for each size
        for i, size in enumerate(self.sizes):
            for j, anchor in enumerate(base_anchors[i]):
                scale = self.scales[j // len(self.ratios)]
                ratio = self.ratios[j % len(self.ratios)]
                self.assert_anchor(anchor, size, scale, ratio)

    def assert_anchor(self, anchor, size, scale, ratio):
        """Assert the area an ratio of a given anchor"""
        # Areas
        actual_area = round(((anchor[2] - anchor[0]) * (anchor[3] - anchor[1])).item())
        expected_area = round((size * scale) ** 2)
        self.assertTrue(abs(expected_area - actual_area) <= 1)
        # Aspect ratios
        actual_ratio = ((anchor[3] - anchor[1]) / (anchor[2] - anchor[0])).item()
        self.assertTrue(abs(ratio - actual_ratio) < 0.1)

    def test_anchors(self):
        """Test that each one of the locations in each feature map has its len(scales) * len(ratios) anchors."""
        images = torch.zeros((5, 3, 128, 128))  # Create 5 "images" of 128 x 128
        anchors = self.anchors(images)  # Generate the anchors
        self.assertEqual(5, anchors.shape[0])
        n_anchors = len(self.scales) * len(self.ratios)  # Number of anchors per location
        # The images will be reduced by the given strides, and then each location will have n_anchors per location
        expected_anchors = 0
        for stride in self.strides:
            height, width = images.shape[2:]
            height, width = height // stride, width // stride
            expected_anchors += height * width * n_anchors
        self.assertEqual(expected_anchors, anchors.shape[1])
        # Test that each anchor has shifted the correct amount and that keeps its correct area and ratio
        for anchors in anchors:
            tested_anchors = 0
            for s, (size, stride) in enumerate(zip(self.sizes, self.strides)):
                height, width = images.shape[2:]
                height, width = height // stride, width // stride
                n_anchors_to_test = height * width * n_anchors
                for i in range(height):
                    for j in range(width):
                        for k in range(n_anchors):
                            actual_index = i * width * n_anchors + j * n_anchors + k + tested_anchors
                            anchor = anchors[actual_index]
                            scale = self.scales[k // len(self.ratios)]
                            ratio = self.ratios[k % len(self.ratios)]
                            # Assert area and the aspect ratio
                            self.assert_anchor(anchor, size, scale, ratio)
                            # Assert shift
                            shift_x = (j * stride) + stride * 0.5
                            shift_y = (i * stride) + stride * 0.5
                            base_anchor = self.anchors.base_anchors[s, k, :]
                            self.assertTrue(abs(base_anchor[0] + shift_x - anchor[0]) < 1)
                            self.assertTrue(abs(base_anchor[2] + shift_x - anchor[2]) < 1)
                            self.assertTrue(abs(base_anchor[1] + shift_y - anchor[1]) < 1)
                            self.assertTrue(abs(base_anchor[3] + shift_y - anchor[3]) < 1)
                tested_anchors += n_anchors_to_test


if __name__ == '__main__':
    unittest.main()
