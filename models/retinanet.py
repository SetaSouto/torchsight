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
        c5_depth = self.backbone.output_depths[0]
        c4_depth = self.backbone.output_depths[1]
        c3_depth = self.backbone.output_depths[2]

        # The paper names the feature maps as P. We start from the last output, the more rich semantically.

        # Conv 1x1 to set features dimension and conv 3x3 to generate feature map
        self.p5_conv1 = nn.Conv2d(c5_depth, features, kernel_size=1, stride=1, padding=0)
        self.p5_conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        # Upsample to sum with c_4
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_conv1 = nn.Conv2d(c4_depth, features, kernel_size=1, stride=1, padding=0)
        self.p4_conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

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

    --- Anchors ---

    Keep in mind that this network uses anchors, so for each location (or "pixel") of the feature map
    it regresses a bounding box for each anchor and predict the class for each bounding box.
    So, for example, if we have a feature map of (10, 10, 2048) and 9 anchors per location we produce
    10 * 10 * 9 bounding boxes.

    The bounding boxes has shape (4) for the x1, y1 (top left corner) and x2, y2 (top right corner).
    Also each bounding box has a vector with the probabilities for the C classes.

    This network uses a Feature Pyramid Network as feature extractor, so its predicts for several feature maps
    of different scales. This is useful to improve the precision over different object scales (very little ones
    and very large ones).

    --- Anchors' sizes ---

    The feature pyramid network (FPN) produces feature maps at different scales, so we use different anchors per scale,
    in the original paper or RetinaNet they use images of size 600 * 600 and the 5 levels of the FPN (P3, ..., P7)
    with anchors with areas of 32 * 32 to 512 * 512.

    Each anchor size is adapted to three different aspect ratios {1:2, 1:1, 2:1} and to three different scales
    {2 ** 0, 2 ** (1/3), 2 ** (2/3)} according to the paper.

    So finally, we have 3 * 3 anchors based on one size, totally 3 * 3 * 5 different anchors for the total network.
    Keep in mind that only 3 * 3 anchors are used per location in the feature map and that the FPN produces 5 feature
    maps, that's why we have 3 * 3 * 5 anchors.

    Example:
    If we have anchors_sizes = [32, 64, 128, 256, 512] then anchors with side 32 pixels are used for the P3 output of
    the FPN, 64 for the P4, ... , 512 for the P7.
    Using the scales {2 ** 0, 2 ** (1/3), 2 ** (2/3)} we can get anchors from 32 pixels of side to 813 pixels of side.

    If you want to use different sizes you can provide the sizes, scales or ratios you can provide them in the
    initialization of the network.

    TODO:
    In training mode returns all the predicted values (all bounding boxes and classes for all the anchors)
    and in evaluation mode applies Non-Maximum suppresion to return only the relevant detections with shape
    (N, 5) where N are the number of detections and 5 are for the x1, y1, x2, y2, class' label.
    """

    def __init__(self,
                 classes,
                 resnet=18,
                 features={'pyramid': 256, 'regression': 256, 'classification': 256},
                 anchors_sizes=[32, 64, 128, 256, 512],
                 anchors_scales=[2, 2 ** (1/3), 2 ** (2/3)],
                 anchors_ratios=[0.5, 1, 2],
                 pretrained=True):
        """Initialize the network.

        Args:
            classes (int): The number of classes to detect.
            resnet (int, optional): The depth of the resnet backbone for the Feature Pyramid Network.
            features (dict, optional): The dict that indicates the features for each module of the network.
            anchors (int, optional): The number of anchors to use per location of the feature map.
            anchors_sizes (sequence, optional): The sizes of the anchors (one side, not area) to use at each different
                scale of the FPN. Must have length 5, for each level of the FPN.
            anchors_scales (sequence, optional): The scales to multiply each anchor size.
            anchors_ratios (sequence, optional): The aspect ratios that the anchors must follow.
            pretrained (bool, optional): If the resnet backbone of the FPN must be pretrained on the ImageNet dataset.
                This pretraining is provided by the torchvision package.
        """
        super(RetinaNet, self).__init__()

        # Generate anchors
        if len(anchors_sizes) != 5:
            raise ValueError('anchors_size must have length 5 to work with the FPN')
        self.anchors = self.generate_anchors(anchors_sizes, anchors_scales, anchors_ratios)
        n_anchors = self.anchors.shape[1]  # The number of anchors per size

        # Modules
        self.fpn = FeaturePyramid(resnet=resnet, features=features['pyramid'], pretrained=pretrained)
        self.regression = Regression(in_channels=features['pyramid'], anchors=n_anchors,
                                     features=features['regression'])
        self.classification = Classification(in_channels=features['pyramid'], classes=classes, anchors=n_anchors,
                                             features=features['classification'])

        # Loss
        # TODO: Add loss and anchors, study code

    @staticmethod
    def generate_anchors(sizes, scales, ratios):
        """Given a sequence of side sizes generate len(scales) *  len(ratios) anchors per size.

        Args:
            anchors_sizes (sequence): Sequence of int that are the different sizes of the anchors.

        Returns:
            torch.Tensor: Tensor with shape (len(anchors_sizes), len(scales) * len(ratios), 4).
        """
        n_anchors = len(scales) * len(ratios)
        scales, ratios = torch.Tensor(scales), torch.Tensor(ratios)
        # First we are going to compute the anchors as center_x, center_y, height, width
        anchors = torch.zeros((len(sizes), n_anchors, 4), dtype=torch.float)
        # Start with height = width = 1
        anchors[:, :, 2:] = torch.Tensor([1., 1.])
        # Scale each anchor to the correspondent size. We use unsqueeze to get sizes with shape (len(sizes), 1, 1)
        # and broadcast to (len(sizes), n_anchors, 4)
        anchors *= sizes.unsqueeze(1).unsqueeze(1)
        # Multiply the height for the aspect ratio. We repeat the ratios len(scales) times to get all the aspect
        # ratios for each scale. We unsqueeze the ratios to get the shape (1, n_anchors) to broadcast to
        # (len(sizes), n_anchors)
        anchors[:, :, 2] *= ratios.repeat(len(scales)).unsqueeze(0)
        # Adjust width and height to match the area size * size
        areas = sizes * sizes  # Shape (len(sizes))
        height, width = anchors[:, :, 2], anchors[:, :, 3]  # Shapes (len(sizes), n_anchors)
        adjustment = torch.sqrt((height * width) / areas.unsqueeze(1))
        anchors[:, :, 2] /= adjustment
        anchors[:, :, 3] /= adjustment
        # Multiply the height and width by the correspondent scale. We repeat the scale len(ratios) times to get
        # one scale for each aspect ratio. So scales has shape (1, n_anchors, 1) and
        # broadcast to (len(sizes), n_anchors, 2) to scale the height and width.
        anchors[:, :, 2:] *= scales.unsqueeze(1).repeat((1, len(ratios))).view(1, -1, 1)

        # Return the anchors but not centered nor with height or width, instead use x1, y1, x2, y2
        height, width = anchors[:, :, 2].clone(), anchors[:, :, 3].clone()
        center_x, center_y = anchors[:, :, 0].clone(), anchors[:, :, 1].clone()

        anchors[:, :, 0] = center_x - (width * 0.5)
        anchors[:, :, 1] = center_y - (height * 0.5)
        anchors[:, :, 2] = center_x + (width * 0.5)
        anchors[:, :, 3] = center_y + (height * 0.5) 

        return anchors

    @staticmethod
    def transform(mean=None, std=None):
        """Transforms the bounding boxes given by the regression model.

        It uses ...
        """

    @staticmethod
    def clip(boxes, batch):
        """Given the boxes predicted for the batch, clip the boxes to fit the width and height.

        This means that if the box has any side outside the dimensions of the images the side is adjusted
        to fit inside the image. For example, if the image has width 800 and the right side of a bounding
        box is at x = 830, then the right side will be x = 800.

        Args:
            boxes (torch.Tensor): A tensor with the parameters for each bounding box.
                Shape:
                    (batch size, number of bounding boxes, 4).
                Parameters:
                    boxes[:, :, 0]: x1. Location of the left side of the box.
                    boxes[:, :, 1]: y1. Location of the top side of the box.
                    boxes[:, :, 2]: x2. Location of the right side of the box.
                    boxes[:, :, 3]: y2. Location of the bottom side of the box.
            batch (torch.Tensor): The batch with the images. Useful to get the width and height of the image.
                Shape:
                    (batch size, channels, width, height)

        Returns:
            torch.Tensor: The clipped bounding boxes with the same shape.
        """
        _, _, height, width = batch.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
