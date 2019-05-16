"""Test the Logo32plus dataset."""
import unittest

import torch

from torchsight.datasets import Logo32plusDataset


class Logo32plusDatasetTest(unittest.TestCase):
    """TestCase for the Logo32Plus dataset."""

    def test_annotations(self):
        """Test that the dataset can load the annotations."""
        dataset = Logo32plusDataset('/home/souto/datasets/logo32plus')
        for annot in dataset.annotations:
            image, boxes, name = annot
            self.assertIsInstance(image, str)
            self.assertIsInstance(boxes, torch.Tensor)
            self.assertIsInstance(name, str)
        self.assertIsInstance(dataset.class_to_label, dict)
        self.assertIsInstance(dataset.label_to_class, dict)


if __name__ == '__main__':
    unittest.main()
