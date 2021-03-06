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

    def __init__(self, in_channels, embedding_size, anchors, features, classes, normalize=False,
                 weighted_bias=False, fixed_bias=None, increase_norm_by=None, prior=None):
        """Initialize the classification module.

        Arguments:
            in_channels (int): The number of channels of the feature map.
            embedding_size (int): Length of the embedding vector to generate.
            anchors (int): Number of anchors per location in the feature map.
            features (int): Number of features in the conv layers that generates the embedding.
            classes (int): The number of classes to detect.
            normalize (bool, optional): Indicate that it must normalize the embeddings.
            weighted_bias (bool, optional): If True it uses bias weights to perform the classification.
            fixed_bias (float, optional): Use a bias for the classification as an hyperparameter.
            increase_norm_by (float, optional): Increase the norm of the classification vectors during
                the classification by this value.
            prior (float): the prior to set in the bias of the classification layer. None will init
                the bias with zeros.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.classes = classes
        self.normalize = normalize

        self.encoder = SubModule(in_channels=in_channels, outputs=embedding_size, anchors=anchors, features=features)

        # Keep track of the generated embeddings, they are populated with the forward method
        self.embeddings = None

        self.sigmoid = nn.Sigmoid()
        self.weights = nn.Parameter(torch.Tensor(embedding_size, classes))

        self.weighted_bias = weighted_bias
        if self.weighted_bias:
            self.bias = nn.Parameter(torch.Tensor(classes))

        self.fixed_bias = fixed_bias

        if self.fixed_bias is not None and self.weighted_bias:
            print('WARN: Using weighted and fixed bias in the classification module, '
                  'this could lead to inconsistent results.')

        self.norm_increaser = increase_norm_by
        self.reset_weights(prior=prior)

    def reset_weights(self, prior=None):
        """Reset and initialize the weights with kaiming uniform and the bias with constant value.

        For the bias you can use a special constant. As at the beginning of the training all the embeddings
        have proability near 0.5 to belong to a class we can set a bias so the prior of the embedding could be
        a given value.
        Usually, in sigmoid based loss functions, is used 1/C the prior for each output as we said that
        any class could be the output.
        By default is zero, as it shows better convergence in the future and less false positives,
        but you can change it to provide this other constant.
        """
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if self.weighted_bias:
            bias = 0
            if prior is not None:
                bias = float(-torch.log(torch.Tensor([(1/prior) - 1])))
            nn.init.constant_(self.bias, bias)

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
            embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)

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
        similarity = torch.matmul(embeddings, self.weights)

        if self.norm_increaser is not None:
            similarity *= self.norm_increaser

        if self.weighted_bias:
            similarity += self.bias

        if self.fixed_bias is not None:
            similarity += self.fixed_bias

        return self.sigmoid(similarity)

    def forward(self, feature_maps):
        """Generate the embeddings based on the feature maps and get thr probability of each one
        to belong to any class.

        Arguments:
            feature_maps (list of torch.Tensor): Feature maps generated by the FPN module.
                Shape of each one: (batch size, channels, height, width)

        Returns:
            torch.Tensor: Tensor with the probability for each anchor to belong to each class.
                Shape:
                    (batch size, feature map's height * width * number of anchors, classes)
        """
        self.embeddings = torch.cat([self.encode(feature_map) for feature_map in feature_maps], dim=1)
        return self.classify(self.embeddings)

    def get_features(self, feature_maps, conv=1):
        """Get the features for each location in the feature map after the "conv" convolution of the encoder.

        This is not used for classification but can be used for feature extraction.

        Arguments:
            feature_maps (list of torch.Tensor): Feature maps generated by the FPN module.
                Shape of each one: (batch size, channels, height, width)
            conv (int): the number of the last convolution of the encoder submodule to apply.
                Accepts: {1, 2, 3, 4, 5}.

        Returns:
            list of torch.Tensor: with shape (batch size, num features, height, width) for each one.
        """
        if conv not in [1, 2, 3, 4, 5]:
            raise ValueError("There is no conv{} layer.".format(conv))

        if conv == 5:
            return [self.encode(feature_map) for feature_map in feature_maps]

        out = [self.encoder.conv1(feature_map) for feature_map in feature_maps]
        out = [self.endoder.act1(x) for x in out]
        if conv == 1:
            return out

        out = [self.encoder.conv2(x) for x in out]
        out = [self.encoder.act2(x) for x in out]
        if conv == 2:
            return out

        out = [self.encoder.conv3(x) for x in out]
        out = [self.encoder.act3(x) for x in out]
        if conv == 3:
            return out

        out = [self.encoder.conv4(x) for x in out]
        out = [self.encoder.act4(x) for x in out]
        return out


class DLDENet(RetinaNet):
    """Deep local directional embeddings net.

    Perform object detection by encoding for each anchor an embedding of the object that must point
    in the same direction as its classification vector.

    Based on the RetinaNet implementation of this package, for more information please see its docs.
    """

    def __init__(self, classes, resnet=18, features=None, anchors=None, fpn_levels=None, embedding_size=512,
                 normalize=False, pretrained=True,
                 device=None, weighted_bias=False, fixed_bias=None, increase_norm_by=None, prior=None):
        """Initialize the network.

        Arguments:
            classes (int): The number of classes to detect.
            resnet (int, optional): The depth of the resnet backbone for the Feature Pyramid Network.
            features (dict, optional): The dict that indicates the features for each module of the network.
                For the default dict please see RetinaNet module.
            anchors (dict, optional): The dict with the 'sizes', 'scales' and 'ratios' sequences to initialize
                the Anchors module. For default values please see RetinaNet module.
            fpn_levels (list of int): The numbers of the layers in the FPN to get their feature maps.
                If None is given it will return all the levels from 3 to 7.
                If some level is not present it won't return that feature map level of the pyramid.
            embedding_size (int, optional): The length of the embedding to generate per anchor.
            normalize (bool, optional): Indicates if the embeddings must be normalized.
            pretrained (bool, optional): If the resnet backbone of the FPN must be pretrained on the ImageNet dataset.
                This pretraining is provided by the torchvision package.
            device (str, optional): The device where the module will run.
            weighted_bias (bool, optional): Use bias weights in the classification module.
            fixed_bias (float, optional): A bias to use as a fixed hyperparameter.
            increase_norm_by (float, optional): Increase the norm of the classification vectors by this value while
                performing the classification step.
            prior (float): the prior to set in the bias of the classification layer. None will init
                the bias with zeros.
        """
        self.embedding_size = embedding_size
        self.normalize = normalize
        self.weighted_bias = weighted_bias
        self.fixed_bias = fixed_bias
        self.increase_norm_by = increase_norm_by
        self.prior = prior
        super().__init__(classes, resnet, features, anchors, fpn_levels, pretrained, device)

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
                                    features=features, classes=classes, normalize=self.normalize,
                                    weighted_bias=self.weighted_bias, fixed_bias=self.fixed_bias,
                                    increase_norm_by=self.increase_norm_by, prior=self.prior)

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

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        """Get an instance of the model from a checkpoint generated with the DLDENetTrainer.

        Arguments:
            checkpoint (str or dict): The path to the checkpoint file or the loaded checkpoint file.
            device (str, optional): The device where to load the model.

        Returns:
            DLDENet: An instance with the weights and hyperparameters got from the checkpoint file.
        """
        device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=device)

        params = checkpoint['hyperparameters']['model']

        model = cls(
            classes=params['classes'],
            resnet=params['resnet'],
            features=params['features'],
            anchors=params['anchors'],
            embedding_size=params['embedding_size'],
            normalize=params['normalize'],
            weighted_bias=params['weighted_bias'],
            pretrained=params['pretrained'],
            device=device
        )
        model.load_state_dict(checkpoint['model'])

        return model

    @classmethod
    def from_checkpoint_with_new_classes(cls, checkpoint, num_classes, device=None):
        """Get an instance of the model from a checkpoint generated with the DLDENet trainer
        but chaning the number of classes of the classification module.

        This can be useful to transfer learning between the models.
        This class method will load all the weights except those of the classification layer.

        Arguments:
            checkpoint (str or dict): path to the checkpoint file or the loaded checkpoint file.
            num_classes (int): number of classes to detect with the model.
            device (str, optional): where to load the model.

        Returns:
            DLDENet: with the weights and hyperparameters of the checkpoint but with a new
                classification layer without pretrained weights.
        """
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=device)
        # Instantiate a new DLDENet
        params = checkpoint['hyperparameters']['model']
        model = cls(
            classes=num_classes,
            resnet=params['resnet'],
            features=params['features'],
            anchors=params['anchors'],
            embedding_size=params['embedding_size'],
            normalize=params['normalize'],
            weighted_bias=params['weighted_bias'],
            pretrained=params['pretrained'],
            device=device
        )
        # Get the state dict of the model
        state_dict = checkpoint['model']
        # Remove the weights of the classification layer
        state_dict.pop('classification.weights')
        state_dict.pop('classification.bias')
        # Load the weights in the model
        model.load_state_dict(state_dict, strict=False)

        return model
