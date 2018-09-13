from torch import nn
from torchvision.models import resnet50


class Resnet50(nn.Module):
    """An abstraction of the Resnet50 architecture for fine tuning.

    Inspired by @mratsim at this issue:
    https://github.com/pytorch/examples/pull/58#issuecomment-305950890
    """

    def __init__(self, num_classes=None, image_size=None, activation=None, requires_grad=True):
        """Initialize the network, sets the features module and the classifier module.

        Keep in mind that the network has an stride of 32. So if you have an image as (3, 320, 320)
        this network generate a feature map as (2048, 10, 10).

        The arguments of this method are only needed if you want to get the classifier too. To
        only get the feature extractor you can omit them.

        Args:
            num_classes (int, optional): The number of classes for output.
            image_size (int, optional): The size of the image to calculate the amount of input
                features for the fully connected layer.
            activation (nn.Module, optional): The activation function for the classifier.
                If none is given does not use any activation function.
            requires_grad (bool, optional): Indicates if the features extractor requires grad or
                not.
        """
        super(Resnet50, self).__init__()
        # This network has an stride of 32
        self.stride = 32
        original_model = resnet50(pretrained=True)
        # Get everything but not the las fully connected layer and the average pool
        # The average pool was thought to get the 7x7 image and return a 1x1 image with
        # 2048 filters. Now we want to get all the channels and the complete feature map.
        # Whats the difference between children() and modules() ?
        # See: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551
        self.features_extractor = nn.Sequential(
            *list(original_model.children())[:-2])
        # Freeze parameters if the features extractor does not requires grad
        if not requires_grad:
            for parameter in self.features_extractor.parameters():
                parameter.requires_grad = requires_grad
        # Set the classifier
        self.activation = activation
        self.classifier = None
        if num_classes and image_size:
            if image_size % self.stride != 0:
                raise Exception("""This network has an stride of 32, so please use an image
                size that is a multiple of 32. Actual image size: {}""".format(image_size))
            reduced_image_size = image_size / self.stride
            in_features = 2048 * reduced_image_size * reduced_image_size
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes))
            # Initialize classifier
            for module in self.classifier:
                nn.init.normal_(module.weight)

    def forward(self, inputs):
        """Forward pass of the network.

        The input must match the size given in the initialization of the network if you
        want that the classifier works.

        Args:
            inputs (torch.Tensor): The input to pass forward in the network with shape
                (batch's size, 3, image's size, image's size).

        Returns:
            torch.Tensor: A tensor with shape (batch size, number of class) if it used
                the classifier or (batch size, 2048, reduced image, reduced image) if
                it's used as feature extractor.
        """
        features = self.features_extractor(inputs)
        if not self.classifier:
            return features
        # Pass trough classifier and activation
        features = features.view(features.shape[0], -1)
        outputs = self.classifier(features)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs
