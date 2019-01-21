"""Focal Loss module.

Loss implemented based on the original paper "Focal Loss For Dense Object Detection":
https://arxiv.org/pdf/1708.02002.pdf
"""
import torch
from torch import nn


class FocalLoss(nn.Module):
    """Loss to penalize the detection of objects."""

    def __init__(self, alpha=0.25, gamma=2.0, iou_thresholds={'background': 0.4, 'object': 0.5}, device=None):
        """Initialize the loss.

        Train as background (minimize all the probabilities of the classes) if the IoU is below the 'background'
        threshold and train with the label of the object if the IoU is over the 'object' threshold.
        Ignore the anchors between both thresholds.

        Args:
            alpha (float): Alpha parameter for the loss.
            gamma (float): Gamma parameter for the loss.
            iou_thresholds (dict): Indicates the thresholds to assign an anchor as background or object.
            device (str, optional): Indicates the device where to run the loss.
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.iou_background = iou_thresholds['background']
        self.iou_object = iou_thresholds['object']
        self.device = device if device else 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
                classification_losses.append(torch.Tensor([0]).to(self.device))
                regression_losses.append(torch.Tensor([0]).to(self.device))
                continue

            iou = self.iou(anchors, annotations)  # (number of anchors, number of annotations)
            iou_max, iou_argmax = iou.max(dim=1)  # (number of anchors)
            # Free memory
            del iou
            # Each anchor is associated to a bounding box. Which one? The one that has bigger iou with the anchor
            assigned_annotations = annotations[iou_argmax, :]  # (number of anchors, 5)
            # Free memory
            del iou_argmax
            # Only train bounding boxes where its base anchor has an iou with an annotation over iou_object threshold
            selected_anchors_objects = iou_max > self.iou_object
            selected_anchors_backgrounds = iou_max < self.iou_background
            n_classes = classifications.shape[1]
            # Free memory
            del iou_max

            # Compute classification loss

            # Create the target tensor. Shape (number anchors, number of classes) where the
            # index of the class for the annotation has a 1 and all the others zero.
            targets = torch.zeros((anchors.shape[0], n_classes)).to(self.device)
            # Get the label for each anchor based on its assigned annotation ant turn it on. Do this only
            # for the assigned anchors.
            targets[selected_anchors_objects, assigned_annotations[selected_anchors_objects, 4].long()] = 1.
            # Generate the alpha factor
            alpha = self.alpha * torch.ones(targets.shape).to(self.device)
            # It must be alpha for the correct label and 1 - alpha for the others
            alpha = torch.where(targets == 1, alpha, 1. - alpha)
            # Generate the focal weight
            focal = torch.where(targets == 1, classifications, 1 - classifications)
            focal = alpha * torch.exp(focal, self.gamma)
            # Get the binary cross entropy
            bce = -(targets * torch.log(classifications) + (1. - targets) * torch.log(1 - classifications))
            # Free memory
            del targets
            # Remove the loss for the not assigned anchors (not background, not object)
            selected_anchors = selected_anchors_backgrounds + selected_anchors_objects
            ignored_anchors = 1 - selected_anchors
            bce[ignored_anchors, :] = 0
            # Append to the classification losses
            classification_losses.append((focal * bce).sum() / selected_anchors.sum())
            # Free memory
            del alpha, focal, bce, ignored_anchors, classifications, selected_anchors_backgrounds

            # Compute regression loss

            # Append zero loss if there is no object
            if not selected_anchors_objects.sum() > 0:
                regression_losses.append(torch.Tensor([0]).to(self.device))
                continue

            # Get the assigned ground truths to each one of the selected anchors to train the regression
            assigned_annotations = assigned_annotations[selected_anchors_objects, :-1]
            # Keep only the regressions for the selected anchors
            regressions = regressions[selected_anchors_objects, :]
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

            # Append the squared error
            regression_losses.append(torch.abs(targets - regressions) ** 2)

        # Return the mean classification loss and the mean regression loss
        return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()

    @staticmethod
    def iou(anchors, annotations):
        """Calculates the Intersection over Union (IoU) between the anchors and the bounding boxes.

        Args:
            anchors (torch.Tensor): The tensor with the base anchors.
                4 values for x1, y1, x2, y2.
                Shape:
                    (total anchors, 4)
            annotations (torch.Tensor): The tensor with the annotations of the image.
                4 values for x1, y1, x2, y2.
                Shape:
                    (number of annotations, 4)

        Returns:
            torch.Tensor: The iou between all the anchors and the annotations.
                Shape:
                    (number of anchors, number of annotations)

        How this work?

        Let's say that d = batch size * width * height and b = number of annotations.

        The beautiful idea behind this function is to unsqueeze the anchors (add one dimension) to change the
        shape from (d) to (d, 1) and with the bounding box with shape (b) the operation broadcast
        to finally get shape (d, b).

        Broadcast:
        PyTorch: https://pytorch.org/docs/stable/notes/broadcasting.html
        Numpy: https://docs.scipy.org/doc/numpy-1.10.4/user/basics.broadcasting.html

        The broadcast always goes from the last dimension to the first, so we have (b) with (d, 1), that is
        to look the tensors as:

        (d, 1)
        (   b)
        ------
        (d, b)

        So the final result is a tensor with shape (d, b).

        How can we read this broadcasting?

        The tensor with shape (d, 1) has "d rows with 1 column", and the vector with shape (b) has "b columns",
        this is the correct way to read the shapes. That's why if you print a tensor with shape (b) you get
        a vector as a row. And if you print a tensor with shape (d, 1) is like a matrix with d rows
        and 1 column. Always we start reading the tensors from the last element of its shape and in the order
        of "columns -> rows -> channels -> batches".

        So the first tensor with shape (d, 1) to match the (d, b) shape must repeat its single element of each
        row to match the column size. Ex:
        [[0],       [[0, 0, 0],
         [1],  -->   [1, 1, 1],
         [2]]        [2, 2, 2]]

        And the second tensor with shape (b) to match the (d, b) shape must repeat the values for each column
        to match the d rows. Ex:
        [3, 4, 5, 6]  --> [[3, 4, 5, 6],
                           [3, 4, 5, 6],
                           [3, 4, 5, 6]]

        Keep in mind this trick.
        Check how these "vectors" broadcast to the (d, b) matrix but with different ways. One repeat the column
        and the other repeat the rows.

        And now we can make calculations all vs all between the d anchors and the b bounding boxes without
        the need of a for loop. That's beautiful and efficient.

        Also, notice that the images are (channels, height, width), totally consequent with this vision on how
        to read the shapes.
        """
        # Shape (number of anchors)
        anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        # Shape (number of annotations)
        annotations_area = (annotations[:, 2] - annotations[:, 0]) * (annotations[:, 3] - annotations[:, 1])

        # Shape (anchors, annotations)
        intersection_x1 = torch.max(anchors[:, 0].unsqueeze(dim=1), annotations[:, 0])
        intersection_y1 = torch.max(anchors[:, 1].unsqueeze(dim=1), annotations[:, 1])
        intersection_x2 = torch.min(anchors[:, 2].unsqueeze(dim=1), annotations[:, 2])
        intersection_y2 = torch.min(anchors[:, 3].unsqueeze(dim=1), annotations[:, 3])

        intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)
        intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)
        intersection_area = intersection_width * intersection_height

        union_area = torch.unsqueeze(anchors_area, dim=1) + annotations_area - intersection_area
        union_area = torch.clamp(union_area, min=1e-8)

        return intersection_area / union_area
