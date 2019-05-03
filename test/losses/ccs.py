"""Tests for the CCS loss."""
import unittest

import torch

from torchsight.losses import CCSLoss
from torchsight.models import Anchors


class TestFocalLoss(unittest.TestCase):
    """Test that the Focal Loss works well."""

    def setUp(self):
        """Initialize the loss."""
        self.loss = CCSLoss(device='cpu')
        self.anchors = Anchors(device='cpu')

    def get_annotations_and_anchors(self):
        """Get a dummy batch with one image of 450x450 with 3 objects."""
        annotations = torch.Tensor([[0., 0., 100., 100., 3.],
                                    [150., 150., 350., 350., 2.],
                                    [400., 400., 450., 450., 1.]])
        anchors = self.anchors(torch.zeros((1, 3, 450, 450)))[0]  # Simulate a batch with an image

        return annotations, anchors

    def get_arguments_for_loss(self, embedding_size=128, n_classes=4):
        """Get a fake batch with anchors, annotations and embeddings (with only ones)"""
        annotations, anchors = self.get_annotations_and_anchors()
        n_anchors = anchors.shape[0]
        embeddings = torch.ones((n_anchors, embedding_size))
        weights = torch.ones((embedding_size, n_classes))

        # Add fake batch dimension
        return anchors.unsqueeze(dim=0), embeddings.unsqueeze(dim=0), weights, annotations.unsqueeze(dim=0)

    def test_minimum_loss(self):
        """Test that with embeddings in the same direction as the weights we get the minimum loss."""
        anchors, embeddings, weights, annotations = self.get_arguments_for_loss()
        embeddings *= 100

        # Add fake batch dimension and compute loss
        loss = self.loss(anchors, embeddings, weights, annotations)

        self.assertEqual(0., float(loss))

    def test_maximum_loss(self):
        """Test that if the embeddings point to another direction we get a big loss."""
        anchors, embeddings, weights, annotations = self.get_arguments_for_loss()

        embeddings *= -1

        loss = self.loss(anchors, embeddings, weights, annotations)

        self.assertEqual(1., float(loss))

    def test_random(self):
        """Test that the loss gives a value between 0 and 1."""
        anchors, embeddings, weights, annotations = self.get_arguments_for_loss()
        embeddings = torch.rand_like(embeddings)
        weights = torch.rand_like(weights)

        loss = self.loss(anchors, embeddings, weights, annotations)

        self.assertGreater(1, float(loss))
        self.assertLess(0, float(loss))


if __name__ == '__main__':
    unittest.main()
