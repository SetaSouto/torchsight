"""The criterion for the weighted DLDENet."""
import torch
from torch import nn

from .ccs import CCSLoss
from .focal import FocalLoss


class DLDENetLoss(nn.Module):
    """Join the CCS and the Focal losses in one single module."""

    def __init__(self, alpha=0.25, gamma=2.0, sigma=3.0, iou_thresholds=None, device=None):
        """Initialize the losses.

        See their corresponding docs for more information.

        Arguments:
            alpha (float): Alpha parameter for the focal loss.
            gamma (float): Gamma parameter for the focal loss.
            sigma (float): Point that defines the change from L1 loss to L2 loss (smooth L1).
            iou_thresholds (dict): Indicates the thresholds to assign an anchor as background or object.
            device (str, optional): Indicates the device where to run the loss.
        """
        super().__init__()

        if iou_thresholds is None:
            iou_thresholds = {'background': 0.4, 'object': 0.5}

        device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.focal = FocalLoss(alpha, gamma, sigma, iou_thresholds, device)
        self.ccs = CCSLoss(iou_thresholds, device)

    def forward(self, anchors, regressions, classifications, annotations, model):
        """Compute the different losses for the batch.

        Arguments:
            anchors (torch.Tensor): The base anchors (without the transformation to adjust the
                bounding boxes).
                Shape:
                    `(batch size, total boxes, 4)`
            regressions (torch.Tensor): The regression values to adjust the anchors to the predicted
                bounding boxes.
                Shape:
                    `(batch size, total boxes, 4)`
            classifications (torch.Tensor): The probabilities for each class at each bounding box.
                Shape:
                    `(batch size, total boxes, number of classes)`
            annotations (torch.Tensor): Ground truth. Tensor with the bounding boxes and the label for
                the object. The values must be x1, y1 (top left corner), x2, y2 (bottom right corner)
                and the last value is the label.
                Shape:
                    `(batch size, maximum objects in any image, 5)`
            model (torch.nn.Module): The DLDENet model with its embeddings and weights.

                Why `maximum objects in any image`? Because if we have more than one image, each image
                could have different amounts of objects inside and have different dimensions in the
                ground truth (dim 1 of the batch). So we could have the maximum amount of objects
                inside any image and then the rest of the images ground truths could be populated
                with -1.0. So if this loss finds a ground truth box populated with -1.0 it understands
                that it was to match the dimensions and have only one tensor and it won't use that
                annotation.

        Returns:
            tuple: A tuple with the tensors with the classification, regression and cosine similarity losses.
        """
        classification, regression = self.focal(anchors, regressions, classifications, annotations)
        similarity = self.ccs(anchors, model.embeddings, model.weights, annotations)

        return classification, regression, similarity
