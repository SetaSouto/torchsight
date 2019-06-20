"""Focal Loss module.

Loss implemented based on the original paper "Focal Loss For Dense Object Detection":
https://arxiv.org/pdf/1708.02002.pdf
"""
import torch
from torch import nn

from ..models import Anchors


class FocalLoss(nn.Module):
    """Loss to penalize the detection of objects."""

    def __init__(self, alpha=0.25, gamma=2.0, sigma=3.0, iou_thresholds=None, soft=False, device=None):
        """Initialize the loss.

        Train as background (minimize all the probabilities of the classes) if the IoU is below the 'background'
        threshold and train with the label of the object if the IoU is over the 'object' threshold.
        Ignore the anchors between both thresholds.

        Args:
            alpha (float): Alpha parameter for the focal loss.
            gamma (float): Gamma parameter for the focal loss.
            sigma (float): Point that defines the change from L1 loss to L2 loss (smooth L1).
            iou_thresholds (dict): Indicates the thresholds to assign an anchor as background or object.
            soft (bool, optional): Apply soft classification loss according to theirs IoU.
            device (str, optional): Indicates the device where to run the loss.
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        if iou_thresholds is None:
            iou_thresholds = {'background': 0.4, 'object': 0.5}
        self.iou_background = iou_thresholds['background']
        self.iou_object = iou_thresholds['object']
        self.soft = soft
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def to(self, device):
        """Move the module and its attributes to the given device.

        Arguments:
            device (str): The device where to move the loss.
        """
        self.device = device
        return super(FocalLoss, self).to(device)

    def forward(self, anchors, regressions, classifications, annotations):
        """Forward pass to get the loss.

        Args:
            anchors (torch.Tensor): The base anchors (without the transformation to adjust the
                bounding boxes).
                Shape:
                    (batch size, total boxes, 4)
            regressions (torch.Tensor): The regression values to adjust the anchors to the predicted
                bounding boxes.
                Shape:
                    (batch size, total boxes, 4)
            classifications (torch.Tensor): The probabilities for each class at each bounding box.
                Shape:
                    (batch size, total boxes, number of classes)
            annotations (torch.Tensor): Ground truth. Tensor with the bounding boxes and the label for
                the object. The values must be x1, y1 (top left corner), x2, y2 (bottom right corner)
                and the last value is the label.
                Shape:
                    (batch size, maximum objects in any image, 5).

                Why maximum objects in any image? Because if we have more than one image, each image
                could have different amounts of objects inside and have different dimensions in the
                ground truth (dim 1 of the batch). So we could have the maximum amount of objects
                inside any image and then the rest of the images ground truths could be populated
                with -1.0. So if this loss finds a ground truth box populated with -1.0 it understands
                that it was to match the dimensions and have only one tensor.

        Returns:
            torch.Tensor: The mean classification loss.
            torch.Tensor: The mean regression loss.
        """
        batch_size = anchors.shape[0]
        batch_anchors = anchors
        batch_regressions = regressions
        batch_classifications = classifications
        batch_annotations = annotations

        classification_losses = []
        regression_losses = []

        for index in range(batch_size):
            anchors = batch_anchors[index]
            regressions = batch_regressions[index]
            classifications = batch_classifications[index]
            annotations = batch_annotations[index]
            # Keep only the real labels
            annotations = annotations[annotations[:, -1] != -1]

            if annotations.shape[0] == 0:
                classification_losses.append(torch.zeros((1,)).mean().to(self.device))
                regression_losses.append(torch.zeros((1,)).mean().to(self.device))
                continue

            # Get assignations of the annotations to the anchors
            # Get the assigned annotations (the i-th assigned annotation is the annotation assigned to the i-th
            # anchor)
            # Get the masks to select the anchors assigned to an object (IoU bigger than iou_object threshold)
            # Get the mask to select the anchors assigned to background (IoU lower than iou_background)
            assignations = Anchors.assign(anchors,
                                          annotations,
                                          thresholds={'object': self.iou_object, 'background': self.iou_background})
            assigned_annotations, selected_anchors_objects, selected_anchors_backgrounds, objects_iou = assignations

            # Compute classification loss

            # Create the target tensor. Shape (number anchors, number of classes) where the
            # index of the class for the annotation has a 1 and all the others zero.
            n_classes = classifications.shape[1]
            targets = torch.zeros((anchors.shape[0], n_classes)).to(self.device)
            # Get the label for each anchor based on its assigned annotation ant turn it on. Do this only
            # for the assigned anchors.
            targets[selected_anchors_objects, assigned_annotations[selected_anchors_objects, 4].long()] = 1.
            # Avoid NaN in log clamping the classifications probabilities
            classifications = classifications.clamp(min=1e-5, max=1 - 1e-5)
            # Generate the alpha factor
            alpha = self.alpha * torch.ones(targets.shape).to(self.device)
            # It must be alpha for the correct label and 1 - alpha for the others
            alpha = torch.where(targets == 1, alpha, 1. - alpha)
            # Generate the focal weight
            focal = torch.where(targets == 1, 1 - classifications, classifications)
            focal = alpha * (focal ** self.gamma)
            # Get the binary cross entropy
            bce = -(targets * torch.log(classifications) + (1. - targets) * torch.log(1 - classifications))
            # Free memory
            del targets
            # Remove the loss for the not assigned anchors (not background, not object)
            selected_anchors = selected_anchors_backgrounds + selected_anchors_objects
            ignored_anchors = 1 - selected_anchors
            bce[ignored_anchors, :] = 0
            # Append to the classification losses
            loss = focal * bce
            if self.soft:
                loss[selected_anchors_objects, :] *= objects_iou.unsqueeze(dim=1)
            loss = loss.sum() / selected_anchors.sum()
            classification_losses.append(loss)
            # Free memory
            del alpha, focal, bce, ignored_anchors, classifications, selected_anchors_backgrounds, loss

            # Compute regression loss

            # Append zero loss if there is no object
            if not selected_anchors_objects.sum() > 0:
                regression_losses.append(torch.zeros((1,)).mean().to(self.device))
                continue

            # Get the assigned ground truths to each one of the selected anchors to train the regression
            assigned_annotations = assigned_annotations[selected_anchors_objects, :-1]  # Shape (n selected anchors, 4)
            # Keep only the regressions for the selected anchors
            regressions = regressions[selected_anchors_objects, :]  # Shape (n selected anchors, 4)
            # Keep only the selected anchors for regression
            anchors = anchors[selected_anchors_objects, :]  # Shape (n selected anchors, 4)
            # Free memory
            del selected_anchors_objects

            # Get the parameters from the anchors
            anchors_widths = anchors[:, 2] - anchors[:, 0]
            anchors_heights = anchors[:, 3] - anchors[:, 1]
            anchors_x = anchors[:, 0] + (anchors_widths / 2)
            anchors_y = anchors[:, 1] + (anchors_heights / 2)

            # Get the parameters from the ground truth
            gt_widths = (assigned_annotations[:, 2] - assigned_annotations[:, 0]).clamp(min=1)
            gt_heights = (assigned_annotations[:, 3] - assigned_annotations[:, 1]).clamp(min=1)
            gt_x = assigned_annotations[:, 0] + (gt_widths / 2)
            gt_y = assigned_annotations[:, 1] + (gt_heights / 2)

            # Generate the targets
            targets_dw = torch.log(gt_widths / anchors_widths)
            targets_dh = torch.log(gt_heights / anchors_heights)
            targets_dx = (gt_x - anchors_x) / anchors_widths
            targets_dy = (gt_y - anchors_y) / anchors_heights

            # Stack the targets as it could come from the regression module
            targets = torch.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim=1)
            regression_diff = torch.abs(targets - regressions)
            # Compute Smooth L1 loss
            regression_loss = torch.where(
                regression_diff <= 1 / (self.sigma**2),
                0.5 * (self.sigma * regression_diff) ** 2,
                regression_diff - (0.5 / (self.sigma ** 2))
            )

            # Append the squared error
            regression_losses.append(regression_loss.mean())

        # Return the mean classification loss and the mean regression loss
        return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()
