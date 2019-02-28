"""Tests for the focal loss."""
import unittest

import torch

from torchsight import metrics
from torchsight.losses import FocalLoss
from torchsight.models import Anchors


class TestFocalLoss(unittest.TestCase):
    """Test that the Focal Loss works well."""

    def __init__(self, *args, **kwargs):
        """Initialize the loss and anchors modules."""
        super(TestFocalLoss, self).__init__(*args, **kwargs)

        self.loss = FocalLoss()
        self.anchors = Anchors()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def test_no_annotations(self):
        """Test that with no annotations the loss gives zero losses."""
        # Simulate a batch of size 5 with 10 anchors and 50 classes
        anchors = torch.zeros((5, 10, 4))
        regressions = torch.zeros((5, 10, 4))
        classifications = torch.zeros((5, 10, 50))
        annotations = torch.ones((5, 5, 5)) * -1  # -1 Means no annotation
        # Get the losses
        classification, regression = self.loss(anchors, regressions, classifications, annotations)
        self.assertEqual(0., float(classification))
        self.assertEqual(0., float(regression))

    def test_correct(self):
        """Test that with correct annotations gives zero loss."""
        # Simulate a batch with size 1, 5 classes, 3 annotations and 5 anchors for an image of size 450x450
        annotations = torch.Tensor([[0., 0., 100., 100., 3.],
                                    [150., 150., 350., 350., 2.],
                                    [400., 400., 450., 450., 1.]]).to(self.device)
        anchors = self.anchors(torch.zeros((1, 3, 450, 450)))[0]  # Simulate a batch with an image
        iou = metrics.iou(anchors, annotations)  # Get the iou to assign the annotations to the anchors
        iou, iou_argmax = torch.max(iou, dim=1)  # Get the max iou for each anchor
        # Get the annotation for each anchor. The annotation for the anchor i is assigned_annotation[i]
        assigned_annotations = annotations[iou_argmax]
        # Get the targets for the regression:
        # The regression, as indicated in the second formula in the Faster R-CNN paper:
        # https://arxiv.org/pdf/1506.01497.pdf
        # must have target values like tx = (x - xa) / wa, ty = (y - ya) / ha, tw = log(w/wa) and th =  log(h/ha)
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        xa = anchors[:, 0] + (wa/2)
        ya = anchors[:, 1] + (ha/2)
        w = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        h = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        x = assigned_annotations[:, 0] + (w/2)
        y = assigned_annotations[:, 1] + (h/2)
        tx = (x - xa) / wa
        ty = (y - ya) / ha
        tw = torch.log(w/wa)
        th = torch.log(h/ha)
        regressions = torch.stack([tx, ty, tw, th], dim=1)
        # Get the classification probs according to their assigned annotations
        classifications = torch.zeros((anchors.shape[0], 5)).to(self.device)
        selected_anchors = iou > self.loss.iou_object  # Anchors below that threshold are background
        classifications[selected_anchors, assigned_annotations[selected_anchors, -1].long()] = 1.
        # Get the losses (unsqueeze to add the batch dimension)
        classification_loss, regression_loss = self.loss(anchors.unsqueeze(0),
                                                         regressions.unsqueeze(0),
                                                         classifications.unsqueeze(0),
                                                         annotations.unsqueeze(0))
        self.assertEqual(0., float(regression_loss))
        self.assertLess(abs(0. - float(classification_loss)), 1e20)
        # Change the labels to turn on all the other classes but not the correct one
        classifications = 1 - classifications
        classification_loss, _ = self.loss(anchors.unsqueeze(0),
                                           regressions.unsqueeze(0),
                                           classifications.unsqueeze(0),
                                           annotations.unsqueeze(0))
        self.assertGreater(float(classification_loss), 43.)
        # TODO: Test with no regression values (box == anchor) and with correct label.


if __name__ == '__main__':
    unittest.main()
