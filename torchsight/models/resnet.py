"""Module that contains ResNet implementation.

ResNet could contains several depths. The paper includes 5 different models
and the model_zoo contains pretrained weights for each one:

- ResNet 18
- ResNet 34
- ResNet 50
- ResNet 101
- ResNet 152

Each one of this different architectures is based on "blocks" that help to
reduce complexity of the network. There are two type of blocks:

- Basic: Only applies two 3x3 convolutions. Used in the 18 and 34 architectures.
- Bottleneck: Applies a 1x1 convolution to reduce the channel size of the feature
    map to a 1/4 (i.e. a feature map with 512 channels is reduced to 128 channels),
    then applies a 3x3 convolution with this reduced channels and finally increase
    the channel dimensions again to the original size using a 1x1 convolution.
    This help to reduce the weights to learn and the complexity of the network.

After each convolution it applies a batch normalization and after each block applies
a "Residual connection" that implies to sum the input of the block to the output of it.
This is helpful to learn identity maps, because if F(x) is the output of the block
the final output is F(x) + x, so if the weights went to zero, the output of the block
is only x. The hypothesis of the authors were that is more easy to learn zero weights
than learning 1 weights.

Finally, an architecture is composed by several "layers" that contains several "blocks".
All architectures has 5 layers, the difference is the kind of block and how many of them
are used in each one. The first layer is only a single convolutional layer with kernel
7x7 with stride 2, so the layers that contains blocks are only 4.

To see the architectures in detail you can go to the Table 1 in the original paper.

Original paper:
https://arxiv.org/pdf/1512.03385.pdf

Heavily inspired by the original code at:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from torch import nn
from torch.utils import model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    It applies two 3x3 convolutions to the input. After each convolution
    applies a batch normalization.

    You can provide a downsample module to downsample the input and sum
    to the output (Residual connection) if not provided it assumes that
    the input has the same dimension of the output.

    This block has no expansion (i.e. = 1), this means that the number of
    channels of the output feature map are the same as the input.
    """

    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        """Initialize the block and set all the modules needed.

        Args:
            in_channels (int): Number of channels of the input feature map.
            channels (int): Number of channels that the block must have.
                Also, this is the number of channels to output.
            stride (int): The stride of the convolutional layers.
            downsample (torch.nn.Module): Module downsample the output.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Any tensor with shape (batch size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the block with shape
                (batch size, channels, height / stride, width / stride)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Applies a convolution of kernel 1x1 to reduce the number of channels from
    in_channels to channels, then applies a 3x3 convolution and finally expand
    the channels to 4 * channels (i.e. expansion = 4) with a 1x1 convolution.

    You can provide a downsample module to downsample the input and sum
    to the output (Residual connection) if not provided it assumes that
    the input has the same dimension of the output.
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Any tensor with shape (batch size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the block with shape
                (batch size, channels * expansion, height / stride, width / stride)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet architecture.

    Implements a ResNet given the type of block and the depth of the layers.

    If you provide the number of classes it can be used as a classifier, if not
    it return the output of each layer starting from the deepest.

    Keep in mind that for the first layer the stride is 2 ** 2 = 4, and the
    consecutive ones are 2 ** 3 = 8, 2 ** 4 = 16, 2 ** 5 = 32.

    So, if you provide an image with shape (3, 800, 800) the output of the last layer
    will be (512 * block.expansion, 25, 25).
    """

    def __init__(self, block, layers, num_classes=None):
        """Initialize the network.

        Args:
            block (torch.nn.Module): Indicates the block to use in the network. Must be
                a BasicBlock or a Bottleneck.
            layers (seq): Sequence to indicate the number of blocks per each layer.
                It must have length 4.
            num_classes (int, optional): If present initialize the architecture as a classifier
                 and append a fully connected layer to map from the feature map to the class
                 probabilities. If not present, the module returns the output of each layer.
        """
        super(ResNet, self).__init__()
        # Set the expansion of the net
        self.expansion = block.expansion
        # The depths of each layer
        depths = [64, 128, 256, 512]
        # in_channels help us to keep track of the number of channels before each block
        self.in_channels = depths[0]

        # Layer 1
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The first layer does not apply stride because we use maxPool with stride 2
        self.layer2 = self._make_layer(block, depths[0], layers[0])
        self.layer3 = self._make_layer(block, depths[1], layers[1], stride=2)
        self.layer4 = self._make_layer(block, depths[2], layers[2], stride=2)
        self.layer5 = self._make_layer(block, depths[3], layers[3], stride=2)

        self.classifier = False
        if num_classes is not None and num_classes > 0:
            # Set the classifier
            self.classifier = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fully = nn.Linear(512 * block.expansion, num_classes)
        else:
            # Set the output's number of channels for each layer, useful to get the output depth outside this module
            # when using it as feature extractor
            self.output_channels = [depth * self.expansion for depth in depths[-3:]]
            self.output_channels.reverse()

        # Initialize network
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        """Creates a layer for the ResNet architecture.

        It uses the given 'block' for the layer and repeat it 'blocks' times.
        Each block expands the number of channels by a factor of block.expansion
        times, so the 'in_channels' for every block after the first is block.expansion
        times the 'channel' amount.

        This method modifies the in_channels attribute of the object to keep track of the
        number of channels before each block.

        Args:
            block (Module): Block class to use as the base block of the layer.
            channels (int): The number of channels that the block must have.
            blocks (int): How many blocks the layer must have.
            stride (int): Stride that the first block must apply. None other block
                applies stride.
        """
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            # Apply a module to reduce the width and height of the input feature map with the given stride
            # or to adjust the number of channels that the first block will receive
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        # Only the first block applies a stride to the input
        layers.append(block(self.in_channels, channels, stride, downsample))
        # Now the in_channels are the output of the block that is the channels times block.expansion
        self.in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the module.

        Pass the input tensor for the layers and has two different outputs depending
        if the module is used as a classifier or not.

        Args:
            x (torch.Tensor): A tensor with shape (batch size, 3, height, width).

        Returns:
            torch.Tensor: If the module is a classifier returns a tensor as (batch size, num_classes).
                If the module is a feature extractor (no num classes given) then returns a tuple
                with the output of the last 3 layers.
                The shapes are:
                    - layer 5: (batch size, 512 * block.expansion, height / 32, width / 32)
                    - layer 4: (batch size, 256 * block.expansion, height / 16, width / 16)
                    - layer 3: (batch size, 128 * block.expansion, height / 8,  width / 8)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer2(x)

        if self.classifier:
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fully(x)
            return x

        output3 = self.layer3(x)
        output4 = self.layer4(output3)
        output5 = self.layer5(output4)

        return output5, output4, output3


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(MODEL_URLS['resnet152']))
    return model
