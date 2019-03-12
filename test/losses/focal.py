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

    def get_loss_arguments(self, classes=5):
        """Return the arguments for the loss.

        Simulate a batch with size 1, 'classes' classes and 3 annotations for an image of size 450x450.

        It returns the anchors, regressions, classifications and annotations for a fake image.
        The classification and the regressions are the "perfect" ones, it must be the minimum loss,
        in theory, zero, but as we clamp the values to avoid NaN it must be a value near zero.

        Returns:
            torch.Tensor: The base anchors for the image.
                Shape:
                    (total anchors, 4)
            torch.Tensor: The regression values to adjust the base anchors.
                Shape:
                    (total anchors, 4)
            torch.Tensor: The classifications probabilities for each class. Each classification value
                has the correct value, i.e. has all the values as 0 but no the correct one that is 1.
                Shape:
                    (total anchors, classes)
            torch.Tensor: The fake annotations for the fake image.
                Shape:
                    (3, 5)
        """
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
        classifications = torch.zeros((anchors.shape[0], classes)).to(self.device)
        selected_anchors = iou > self.loss.iou_object  # Anchors below that threshold are background
        classifications[selected_anchors, assigned_annotations[selected_anchors, -1].long()] = 1.

        # Unsqueeze to simulate the batch size of 1
        return (tensor.unsqueeze(0) for tensor in [anchors, regressions, classifications, annotations])

    def test_correct(self):
        """Test that with correct annotations gives zero loss."""
        arguments = self.get_loss_arguments()
        classification_loss, regression_loss = self.loss(*arguments)
        self.assertEqual(0., float(regression_loss))
        classification_loss = float(classification_loss)
        self.assertLess(abs(0. - classification_loss), 1e20)

    def test_convergence(self):
        """Test that the loss has the structure to take the classification values to the correct one
        and it does not have a local minimum.

        How can we do this? By taking the perfect classification and increasing the classification score
        by a little value to all the other non-correct labels. The loss always must increase.
        """
        classes = 80
        print('Using {} classes.'.format(classes))
        anchors, regressions, classifications, annotations = self.get_loss_arguments(classes=classes)
        # Take the MAXIMUM value that the loss could be theoretically by changing all the labels.
        # Change the labels to turn on all the other labels but not the correct one
        classifications_test = 1 - classifications
        classification_loss, _ = self.loss(anchors, regressions, classifications_test, annotations)
        classification_loss = float(classification_loss)
        self.assertGreater(classification_loss, 43.)
        # Iterate increasing the bad labels
        increase_by = 0.1
        last_loss = float(self.loss(anchors, regressions, classifications, annotations)[0])
        mask = classifications != 1
        classifications_test = classifications.clone()
        print('Increasing bad labels:')
        for _ in range(int(1 / increase_by)):
            classifications_test[mask] += increase_by
            loss = float(self.loss(anchors, regressions, classifications_test, annotations)[0])
            current_prob = float(classifications_test[mask][0])
            print('[Current prob {:.7f}] [Last {:.7f}] [Actual {:.7f}]'.format(current_prob, last_loss, loss))
            self.assertLess(last_loss, loss)
            last_loss = loss

        # Check if the loss decreases if the correct ones increases from zero to one
        print('------------------')
        print('Increasing correct label:')
        mask = classifications == 1
        classifications_test = classifications.clone()
        classifications_test[mask] = 0
        last_loss = float(self.loss(anchors, regressions, classifications_test, annotations)[0])
        for _ in range(int(1 / increase_by)):
            classifications_test[mask] += increase_by
            loss = float(self.loss(anchors, regressions, classifications_test, annotations)[0])
            current_prob = float(classifications_test[mask][0])
            print('[Current prob {:.7f}] [Last {:.7f}] [Actual {:.7f}]'.format(current_prob, last_loss, loss))
            self.assertGreater(last_loss, loss)
            last_loss = loss


if __name__ == '__main__':
    unittest.main()
