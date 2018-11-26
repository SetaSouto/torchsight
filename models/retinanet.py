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
"""
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
        c5_depth = 512 * self.backbone.expansion
        c4_depth = 256 * self.backbone.expansion
        c3_depth = 128 * self.backbone.expansion

        # The paper names the feature maps as P

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
