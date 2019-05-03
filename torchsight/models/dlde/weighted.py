"""Weighted implementation of the DLDENet.

The main difference with the tracked one is that this version does not do any track of the mean
of the classes, instead it uses normal weight to perform the classification of the object.

The other version does a normalization of the embeddings and the means that perform the classification
doing cosine similarity and with a modified sigmoid. As shown in the paper
[One-shot Face Recognition by Promoting Underrepresented Classes](https://arxiv.org/pdf/1707.05574.pdf)
this could lead poor performance in the classification.

An idea to clarify why the classification vectors (in the other version called 'mean' because it was
the mean of the embeddings of the classes) must have different norms is because the different classes
could have different intravariance, and with a fixed (modified) sigmoid this could not be expressed.

But is also true that if we have a few samples for a given class the variance is also low and that is
reflected in one-shot or few-shot classification papers' results.

Taking the idea of the paper to do promotion of the underrepresented classes we are going to add to the
loss the necessary conditions to fit what we need:

- That the embeddings goes in the same direction as the classification weight.
- The classification weights could have any norm (not only unit norm).
- Promote the norm of the underrepresented classes.
"""
import math

import torch
from torch import nn

from ..retinanet import RetinaNet, SubModule


class ClassificationModule(nn.Module):
    """The module that performs the classification of the objects.

    It receives the feature pyramid from the backbone network, encode the embeddings and perform the classification.

    It has the parameters to perform the classification simply by doing cosine similarity and then applied a sigmoid.
    """

    def __init__(self, in_channels, embedding_size, anchors, features, classes, normalize=False):
        """Initialize the classification module.

        Arguments:
            in_channels (int): The number of channels of the feature map.
            embedding_size (int): Length of the embedding vector to generate.
            anchors (int): Number of anchors per location in the feature map.
            features (int): Number of features in the conv layers that generates the embedding.
            classes (int): The number of classes to detect.
            normalize (bool, optional): Indicate that it must normalize the embeddings.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.normalize = normalize

        self.encoder = SubModule(in_channels=in_channels, outputs=embedding_size, anchors=anchors, features=features)

        # Keep track of the generated embeddings, they are populated with the forward method
        self.embeddings = None

        self.sigmoid = nn.Sigmoid()
        self.weights = nn.Parameter(torch.Tensor(embedding_size, classes))
        self.reset_weights()

    def reset_weights(self):
        """Reset and initialize with kaiming normal the weights."""
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def encode(self, feature_map):
        """Generate the embeddings for the given feature map.

        Arguments:
            feature_map (torch.Tensor): The features to use to generate the embeddings.
                Shape:
                    (batch size, number of features, feature map's height, width)

        Returns:
            torch.Tensor: The embedding for each anchor for each location in the feature map.
                Shape:
                    (batch size, number of total anchors, embedding size)
        """
        batch_size = feature_map.shape[0]
        # Shape (batch size, number of anchors per location * embedding size, height, width)
        embeddings = self.encoder(feature_map)
        # Move the embeddings to the last dimension
        embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
        # Shape (batch size, number of total anchors, embedding size)
        embeddings = embeddings.view(batch_size, -1, self.embedding_size)

        if self.normalize:
            embeddings /= embeddings.norm(dim=2, keepdim=True)

        return embeddings

    def classify(self, embeddings):
        """Get the probability for each embedding to below to each class.

        Compute the cosine similarity between each embedding and each class' weights and return
        the sigmoid applied over the similarities to get probabilities.

        Arguments:
            embeddings (torch.Tensor): All the embeddings generated.
                Shape:
                    (batch size, total embeddings per image, embedding size)

        Returns:
            torch.Tensor: The probabilities for each embedding.
                Shape:
                    (batch size, total embeddings, number of classes)
        """
        return self.sigmoid(torch.matmul(embeddings, self.weights))

    def forward(self, feature_maps):
        """Generate the embeddings based on the feature maps and get thr probability of each one
        to belong to any class.

        Arguments:
            feature_maps (torch.Tensor): Feature maps generated by the FPN module.
                Shape:
                    (batch size, channels, height, width)

        Returns:
            torch.Tensor: Tensor with the probability for each anchor to belong to each class.
                Shape:
                    (batch size, feature map's height * width * number of anchors, classes)
        """
        self.embeddings = torch.cat([self.encode(feature_map) for feature_map in feature_maps], dim=1)
        return self.classify(self.embeddings)


class DLDENet(RetinaNet):
    """Deep local directional embeddings net.

    Perform object detection by encoding for each anchor an embedding of the object that must point
    in the same direction as its classification vector.

    Based on the RetinaNet implementation of this package, for more information please see its docs.
    """

    def __init__(self, classes, resnet=18, features=None, anchors=None, embedding_size=512, pretrained=True,
                 device=None):
        """Initialize the network.

        Arguments:
            classes (int): The number of classes to detect.
            resnet (int, optional): The depth of the resnet backbone for the Feature Pyramid Network.
            features (dict, optional): The dict that indicates the features for each module of the network.
                For the default dict please see RetinaNet module.
            anchors (dict, optional): The dict with the 'sizes', 'scales' and 'ratios' sequences to initialize
                the Anchors module. For default values please see RetinaNet module.
            embedding_size (int, optional): The length of the embedding to generate per anchor.
            pretrained (bool, optional): If the resnet backbone of the FPN must be pretrained on the ImageNet dataset.
                This pretraining is provided by the torchvision package.
            device (str, optional): The device where the module will run.
        """
        self.embedding_size = embedding_size
        super().__init__(classes, resnet, features, anchors, pretrained, device)

    def get_classification_module(self, in_channels, classes, anchors, features):
        """Get the classification module according to this implementation.

        See __init__ method in RetinaNet class for more information.

        Arguments:
            in_channels (int): The number of channels of the feature map.
            classes (int): Indicates the number of classes to predict.
            anchors (int, optional): The number of anchors per location in the feature map.
            features (int, optional): Indicates the number of inner features that the conv layers must have.

        Returns:
            ClassificationModule: The module for classification.
        """
        return ClassificationModule(in_channels=in_channels, embedding_size=self.embedding_size, anchors=anchors,
                                    features=features, classes=classes)

    def classify(self, feature_maps):
        """Perform the classification of the feature maps.

        We override the original RetinaNet classification method because now we need
        to generate all the embeddings first and then compute the probs to keep track
        of all the embeddings and not only the last one in the for loop.

        Arguments:
            tuple: A tuple with the feature maps generated by the FPN backbone.

        Returns:
            torch.Tensor: The classification probability for each anchor.
                Shape:
                    `(batch size, number of anchors, number of classes)`
        """
        return self.classification(feature_maps)
