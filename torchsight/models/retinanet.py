"""Retinanet module.

This module contains an implementation of each one of the modules needed in the
Retinanet architecture.

Retinanet original paper:
https://arxiv.org/pdf/1708.02002.pdf

- Feature pyramid: Based on a ResNet backbone, takes the output of each layer
    and generate a pyramid of the features.
    Original paper:
    https://arxiv.org/pdf/1612.03144.pdf

- Regression: A module to compute the regressions for each bounding box.
- Classification: A module to classify the class of each bounding box.

This code is heavily inspired by yhenon:
https://github.com/yhenon/pytorch-retinanet
"""
import torch
from torch import nn

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .anchors import Anchors


class FeaturePyramid(nn.Module):
    """Feature pyramid network.

    It takes the natural architecture of a convolutional network to generate a pyramid
    of features with little extra effort.

    It merges the outputs of each one of the ResNet layers with the previous one to give
    more semantically meaning (the deepest the layer the more semantic meaning it has)
    and the strong localization features of the first layers.

    For more information please read the original paper:
    https://arxiv.org/pdf/1612.03144.pdf

    This implementation also used the exact provided at RetinaNet paper:
    https://arxiv.org/pdf/1708.02002.pdf

    Each time that a decision in the code mention "the paper" is the RetinaNet paper.
    """

    strides = [8, 16, 32, 64, 128]  # The stride applied to obtain each output

    def __init__(self, resnet=18, features=256, pretrained=True):
        """Initialize the network.

        It init a ResNet with the given depth and use 'features' as the number of
        channels for each feature map.

        Args:
            resnet (int): Indicates the depth of the resnet backbone.
            features (int): Indicates the depth (number of channels) of each feature map
                of the pyramid.
            pretrained (bool): Indicates if the backbone must be pretrained.
        """
        super(FeaturePyramid, self).__init__()

        if resnet == 18:
            self.backbone = resnet18(pretrained)
        elif resnet == 34:
            self.backbone = resnet34(pretrained)
        elif resnet == 50:
            self.backbone = resnet50(pretrained)
        elif resnet == 101:
            self.backbone = resnet101(pretrained)
        elif resnet == 152:
            self.backbone = resnet152(pretrained)
        else:
            raise ValueError('Invalid ResNet depth: {}'.format(resnet))

        # The paper names the output for each layer a C_x where x is the number of the layer
        # ResNet output feature maps has this depths for each layer
        c5_depth = self.backbone.output_channels[0]
        c4_depth = self.backbone.output_channels[1]
        c3_depth = self.backbone.output_channels[2]

        # The paper names the feature maps as P. We start from the last output, the more rich semantically.

        # Conv 1x1 to set features dimension and conv 3x3 to generate feature map
        self.p5_conv1 = nn.Conv2d(c5_depth, features, kernel_size=1, stride=1, padding=0)
        self.p5_conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        # Upsample to sum with c_4
        self.p5_upsample = lambda input: nn.functional.interpolate(input, scale_factor=2, mode='nearest')

        self.p4_conv1 = nn.Conv2d(c4_depth, features, kernel_size=1, stride=1, padding=0)
        self.p4_conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.p4_upsample = lambda input: nn.functional.interpolate(input, scale_factor=2, mode='nearest')

        # We don't need to upsample c3 because we are not using c2
        self.p3_conv1 = nn.Conv2d(c3_depth, features, kernel_size=1, stride=1, padding=0)
        self.p3_conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

        # In the paper the generate also a p6 and p7 feature maps

        # p6 is obtained via a 3x3 convolution with stride 2 over c5
        self.p6_conv = nn.Conv2d(c5_depth, features, kernel_size=3, stride=2, padding=1)

        # p7 is computed by applying ReLU followed with by a 3x3 convolution with stride 2 on p6
        self.p7_relu = nn.ReLU()
        self.p7_conv = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

    def forward(self, images):
        """Generate the Feature Pyramid given the images as tensor.

        Args:
            images (torch.Tensor): A tensor with shape (batch size, 3, width, height).

        Returns:
            tuple: A tuple with the output for each level from 3 to 7.
                Shapes:
                - p3: (batch size, features, width / 8, height / 8)
                - p4: (batch size, features, width / 16, height / 16)
                - p5: (batch size, features, width / 32, height / 32)
                - p6: (batch size, features, width / 64, height / 64)
                - p7: (batch size, features, width / 128, height / 128)
        """
        # Bottom-up pathway with resnet backbone
        c5, c4, c3 = self.backbone(images)

        # Top-down pathway and lateral connections
        p5 = self.p5_conv1(c5)
        p5_upsampled = self.p5_upsample(p5)
        p5 = self.p5_conv2(p5)

        p4 = self.p4_conv1(c4)
        p4 = p4 + p5_upsampled
        p4_upsampled = self.p4_upsample(p4)
        p4 = self.p4_conv2(p4)

        p3 = self.p3_conv1(c3)
        p3 = p3 + p4_upsampled
        p3 = self.p3_conv2(p3)

        p6 = self.p6_conv(c5)

        p7 = self.p7_relu(p6)
        p7 = self.p7_conv(p7)

        return p3, p4, p5, p6, p7


class SubModule(nn.Module):
    """Base class for the regression and classification submodules."""

    def __init__(self, in_channels, outputs, anchors=9, features=256):
        """Initialize the components of the network.

        Args:
            in_channels (int): Indicates the number of features (or channels) of the feature map.
            outputs (int): The number of outputs per anchor.
            anchors (int, optional): Indicates the number of anchors per location in the feature map.
            features (int, optional): Indicates the number of features that the conv layers must have.
        """
        super(SubModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.last_conv = nn.Conv2d(features, outputs * anchors, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_map):
        """Generates the outputs for each anchor and location in the feature map.

        Args:
            feature_map (torch.Tensor): A tensor with shape (batch size, in_channels, width, height).

        Returns:
            torch.Tensor: The tensor with outputs values for each location and anchor in the feature map.
                Shape:
                    (batch size, outputs * anchors, width, height)
        """
        out = self.conv1(feature_map)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.last_conv(out)

        return out


class Regression(SubModule):
    """Regression submodule of RetinaNet.

    It generates, given a feature map, a tensor with the values of regression for each
    anchor.
    """

    def __init__(self, in_channels, anchors=9, features=256):
        """Initialize the components of the network.

        Args:
            in_channels (int): Indicates the number of features (or channels) of the feature map.
            anchors (int, optional): Indicates the number of anchors (i.e. bounding boxes) per location
                in the feature map.
            features (int, optional): Indicates the number of features that the conv layers must have.
        """
        super(Regression, self).__init__(in_channels, outputs=4, anchors=anchors, features=features)

    def forward(self, feature_map):
        """Generates the bounding box regression for each anchor and location in the feature map.

        Args:
            feature_map (torch.Tensor): A tensor with shape (batch size, in_channels, height, width).

        Returns:
            torch.Tensor: The tensor with bounding boxes values for the feature map.
                Shape:
                    (batch size, height * width * number of anchors, 4)
        """
        out = super(Regression, self).forward(feature_map)

        # Now, out has shape (batch size, 4 * number of anchors, width, height)
        out = out.permute(0, 2, 3, 1).contiguous()  # Set regression values as the last dimension
        return out.view(out.shape[0], -1, 4)  # Change the shape of the tensor


class Classification(SubModule):
    """Classification submodule of RetinaNet.

    It generates, given a feature map, a tensor with the probability of each class.
    """

    def __init__(self, in_channels, classes, anchors=9, features=256):
        """Initialize the network.

        Args:
            in_channels (int): The number of channels of the feature map.
            classes (int): Indicates the number of classes to predict.
            anchors (int, optional): The number of anchors per location in the feature map.
            features (int, optional): Indicates the number of inner features that the conv layers must have.
        """
        super(Classification, self).__init__(in_channels, outputs=classes, anchors=anchors, features=features)

        self.classes = classes
        self.activation = nn.Sigmoid()

    def forward(self, feature_map):
        """Generates the probabilities for each class for each anchor for each location in the feature map.

        Args:
            feature_map (torch.Tensor): A tensor with shape (batch size, in_channels, height, width).

        Returns:
            torch.Tensor: The tensour with the probability of each class for each anchor and location in the
                feature map. Shape: (batch size, height * width * anchors, classes)
        """
        out = super(Classification, self).forward(feature_map)
        out = self.activation(out)

        # Now out has shape (batch size, classes * anchors, height, width)
        out = out.permute(0, 2, 3, 1).contiguous()  # Move the outputs to the last dim
        return out.view(out.shape[0], -1, self.classes)


class RetinaNet(nn.Module):
    """RetinaNet network.

    This Network is for object detection, so its outputs are the regressions
    for the bounding boxes for each anchor and the probabilities for each class for each anchor.

    In training mode returns all the predicted values (all bounding boxes and classes for all the anchors)
    and in evaluation mode applies Non-Maximum suppresion to return only the relevant detections with shape
    (N, 5) where N are the number of detections and 5 are for the x1, y1, x2, y2, class' label.
    """

    def __init__(self, classes, resnet=18, features=None, anchors=None, pretrained=True, device=None):
        """Initialize the network.

        Arguments:
            classes (int): The number of classes to detect.
            resnet (int, optional): The depth of the resnet backbone for the Feature Pyramid Network.
            features (dict, optional): The dict that indicates the features for each module of the network.
            anchors (dict, optional): The dict with the 'sizes', 'scales' and 'ratios' sequences to initialize
                the Anchors module.
            pretrained (bool, optional): If the resnet backbone of the FPN must be pretrained on the ImageNet dataset.
                This pretraining is provided by the torchvision package.
            device (str, optional): The device where the module will run.
        """
        super(RetinaNet, self).__init__()

        if features is None:
            features = {'pyramid': 256, 'regression': 256, 'classification': 256}
        if anchors is None:
            anchors = {'sizes': [32, 64, 128, 256, 512],
                       'scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)],
                       'ratios': [0.5, 1, 2]}

        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Modules
        self.fpn = FeaturePyramid(resnet=resnet, features=features['pyramid'], pretrained=pretrained)

        if len(anchors['sizes']) != 5:
            raise ValueError('anchors_size must have length 5 to work with the FPN')
        self.anchors = Anchors(anchors['sizes'], anchors['scales'], anchors['ratios'],
                               self.fpn.strides, device=self.device)

        self.regression = Regression(in_channels=features['pyramid'],
                                     anchors=self.anchors.n_anchors,
                                     features=features['regression'])
        self.classification = self.get_classification_module(in_channels=features['pyramid'],
                                                             classes=classes,
                                                             anchors=self.anchors.n_anchors,
                                                             features=features['classification'])

        # Set the base threshold for evaluating mode
        self.threshold = 0.6
        self.iou_threshold = 0.5
        self.evaluate_loss = False

    def to(self, device):
        """Move the network to the given device and update the Anchors module that also needs to change
        its device to anchors generation.

        Arguments:
            device (str): The device to where to move the network. Example: 'cpu' or 'cuda:0'.
        """
        self.device = device
        self.anchors.to(device)
        return super(RetinaNet, self).to(device)

    def get_classification_module(self, in_channels, classes, anchors, features):
        """Get the classification module to generate the probability of each anchor
        to be any class.

        Why to do a getter? Because this allow to override this method and compute the
        classification with another method.

        Arguments:
            in_channels (int): The number of channels of the feature map.
            classes (int): Indicates the number of classes to predict.
            anchors (int, optional): The number of anchors per location in the feature map.
            features (int, optional): Indicates the number of inner features that the conv layers must have.

        Returns:
            torch.nn.Module: The classification module for the net.
        """
        return Classification(in_channels=in_channels, classes=classes, anchors=anchors, features=features)

    def eval(self, threshold=None, iou_threshold=None, loss=False):
        """Set the model in the evaluation mode. Keep only bounding boxes with predictions with score
        over threshold.

        Args:
            threshold (float): The threshold to keep only bounding boxes with a class' probability over it.
            iou_threshold (float): If two bounding boxes has Intersection Over Union more than this
                threshold they are detecting the same object.
            loss (bool): If true the forward pass does not do NMS and return the same output as if it is
                training. Why? To compute the loss over the validation dataset for example.
        """
        if threshold is not None:
            self.threshold = threshold

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold

        self.evaluate_loss = loss

        return super(RetinaNet, self).eval()

    def transform(self, images, anchors, regressions, classifications):
        """Transform, clip and apply NMS to the predictions.

        Arguments:
            images (torch.Tensor): The images that passed through the network. Useful for clipping
                the predicted bounding boxes.
                Shape:
                    (batch size, channels, height, width)
            anchors (torch.Tensor): The base anchors generated for the images.
                Shape:
                    (batch size, amount of anchors, 4)
            regressions (torch.Tensor): The regression values to transform and adjust the bounding boxes.
                Shape:
                    (batch size, amount of anchors, 4)
            classifications (torch.Tensor): The probabilities for each class for each anchor.
                Shape:
                    (batch size, amount of anchors, amount of classes)

        Returns:
            sequence: A sequence of (bounding boxes, classifications) for each image.
                Bounding boxes: Tensor with shape (total predictions, 4).
                Classifications: Tensor with shape (total predictions, classes).
        """
        bounding_boxes = self.anchors.transform(anchors, regressions).detach()
        del regressions
        # Clip the boxes to fit in the image
        bounding_boxes = self.anchors.clip(images, bounding_boxes).detach()
        # Generate a sequence of (bounding_boxes, classifications) for each image
        return [self.nms(bounding_boxes[index], classifications[index]) for index in range(images.shape[0])]

    def classify(self, feature_maps):
        """Perform the classification of the feature maps.

        Arguments:
            tuple: A tuple with the feature maps generated by the FPN backbone.

        Returns:
            torch.Tensor: The classification probability for each anchor.
                Shape:
                    `(batch size, number of anchors, number of classes)`
        """
        return torch.cat([self.classification(feature_map) for feature_map in feature_maps], dim=1)

    def forward(self, images):
        """Forward pass of the network. Returns the anchors and the probability for each class per anchor.

        In training mode (calling `model.train()`) it returns all the anchors ans classes' probabilities
        but in evaluation mode (calling `model.eval()`) it applies Non-Maximum Supresion to keep only
        the predictions that do not collide.

        On evaluation mode we cannot return only two tensors (bounding boxes and classifications) because
        different images could have different amounts of predictions over the threshold so we cannot keep
        all them in a single tensor.
        To avoid this problem in evaluation mode it returns a sequence of (bounding_boxes, classifications)
        for each image.

        Args:
            images (torch.Tensor): Tensor with the batch of images.
                Shape:
                    (batch size, channels, height, width)

        Returns:
            In training mode:

            torch.Tensor: A tensor with the base anchors.
                Shape:
                    (batch size, total anchors, 4)
            torch.Tensor: A tensor with the regression values to adjust the anchors to a bounding box.
                Shape:
                    (batch size, total anchors, 4)
            torch.Tensor: A tensor with the probability of each class for each anchor.
                Shape:
                    (batch size, total anchors, number of classes)

            In evaluation mode:

            sequence: A sequence of (bounding boxes, classifications) for each image.
                Bounding boxes: Tensor with shape (total predictions, 4).
                Classifications: Tensor with shape (total predictions, classes).
        """
        feature_maps = self.fpn(images)
        # Get regressions and classifications values with shape (batch size, total anchors, 4)
        regressions = torch.cat([self.regression(feature_map) for feature_map in feature_maps], dim=1)
        classifications = self.classify(feature_maps)
        del feature_maps
        # Transform the anchors to bounding boxes
        anchors = self.anchors(images)

        if self.training:
            return anchors, regressions, classifications
        if self.evaluate_loss:
            return anchors.detach(), regressions.detach(), classifications.detach()

        anchors, regressions, classifications = anchors.detach(), regressions.detach(), classifications.detach()
        return self.transform(images, anchors, regressions, classifications)

    def nms(self, boxes, classifications):
        """Apply Non-Maximum Suppression over the detections to remove bounding boxes that are detecting
        the same object.

        Args:
            boxes (torch.Tensor): Tensor with the bounding boxes.
                Shape:
                    (total anchors, 4)
            classifications (torch.Tensor): Tensor with the scores for each class for each anchor.
                Shape:
                    (total anchors, number of classes)

        Returns:
            torch.Tensor: The bounding boxes to keep.
            torch.Tensor: The probabilities for each class for each bounding box keeped.
        """
        # Get the max score of any class for each anchor
        scores, _ = classifications.max(dim=1)  # Shape (total anchors,)
        # Keep only the bounding boxes and classifications over the threshold
        scores_over_threshold = scores > self.threshold
        boxes = boxes[scores_over_threshold, :]
        classifications = classifications[scores_over_threshold, :]

        if not classifications.shape[0] > 0:
            return torch.zeros((0)).to(self.device), torch.zeros((0)).to(self.device)

        # Update the scores to keep only the keeped boxes
        scores = classifications.max(dim=1)[0]

        # If there aren't detections return empty
        if boxes.shape[0] == 0:
            return torch.zeros(0).to(self.device)

        # Get the numpy version
        # was_cuda = detections.is_cuda
        # detections = detections.cpu().numpy()

        # Start the picked indexes list empty
        picked = []

        # Get the coordinates
        x1 = boxes[:, 0].detach()
        y1 = boxes[:, 1].detach()
        x2 = boxes[:, 2].detach()
        y2 = boxes[:, 3].detach()

        # Compute the area of the bounding boxes
        areas = ((x2 - x1 + 1) * (y2 - y1 + 1)).detach()

        # Get the indexes of the detections sorted by score (lowest score first)
        _, indexes = scores.sort()
        del scores

        while indexes.shape[0] > 1:
            # Take the last index (highest score) and add it to the picked
            actual = int(indexes[-1])
            picked.append(actual)

            # We need to find the overlap of the bounding boxes with the actual picked bounding box

            # Find the largest (more to the bottom-right) (x,y) coordinates for the start
            # (top-left) of the bounding box between the actual and all of the others with lower score
            xx1 = torch.max(x1[actual], x1[indexes[:-1]])
            yy1 = torch.max(y1[actual], y1[indexes[:-1]])
            # Find the smallest (more to the top-left) (x,y) coordinates for the end (bottom-right)
            # of the bounding box
            xx2 = torch.min(x2[actual], x2[indexes[:-1]])
            yy2 = torch.min(y2[actual], y2[indexes[:-1]])

            # Compute width and height to compute the intersection over union
            w = torch.max(torch.Tensor([0]).to(self.device), xx2 - xx1 + 1)
            h = torch.max(torch.Tensor([0]).to(self.device), yy2 - yy1 + 1)
            del xx1, yy1, xx2, yy2
            intersection = (w * h)
            del h, w
            union = areas[actual] + areas[indexes[:-1]] - intersection
            iou = intersection / union

            # Delete the last index
            indexes = indexes[:-1]
            # Keep only the indexes that has overlap lower than the threshold
            indexes = indexes[iou < self.iou_threshold].detach()

        # Return the filtered bounding boxes and classifications
        return boxes[picked].detach(), classifications[picked].detach()
