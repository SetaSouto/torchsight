"""Module with a dummy object detector using a ResNet."""
import torch

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResnetDetector(torch.nn.Module):
    """A dummy detector based on the features extracted from a pretrained ResNet.

    The model generates a feature map based on an image.
    Let's call 'embedding' all the features for a location in the feature map.
    Using a pooling strategy we can reduce the embedding size and get an embedding
    over a kernel size in the feature map.

    So, for example, if we have an image of 512x512, the ResNet has an stride of 32,
    we get a feature map of 16x16 locations. Using a ResNet50, there are 2048 features
    generated per location. We can reduce them to 256 by pooling them.
    Also, we can apply kernels of size 2x2, 4x4 and 8x8 to get other embeddings for
    "bigger" objects.
    """

    def __init__(self, resnet=18, dim=256, pool='avg', kernels=None):
        """Initialize the model.

        Arguments:
            resnet (int, optional): The ResNet to use as feature extractor.
            dim (int, optional): The dimension of the embeddings to generate.
            pool (str, optional): The pool strategy to use. Options: 'avg' or 'max'.
            kernels (list of int, optional): The size of the kernels to use.
        """
        super().__init__()

        if resnet == 18:
            self.resnet = resnet18(pretrained=True)
        elif resnet == 34:
            self.resnet = resnet34(pretrained=True)
        elif resnet == 50:
            self.resnet = resnet50(pretrained=True)
        elif resnet == 101:
            self.resnet = resnet101(pretrained=True)
        elif resnet == 152:
            self.resnet = resnet152(pretrained=True)
        else:
            raise ValueError('There is no resnet "{}"'.format(resnet))

        if pool not in ['avg', 'max']:
            raise ValueError('There is no "{}" pool. Availables: {}'.format(pool, ['avg', 'max']))

        self.dim = dim
        self.pool = pool
        self.kernels = kernels if kernels is not None else [2, 4, 8]

        if pool == 'avg':
            self.pools = [torch.nn.AvgPool2d(k) for k in kernels]
        if pool == 'max':
            self.pools = [torch.nn.MaxPool2d(k) for k in kernels]

    def forward(self, images):
        """Get the embeddings and bounding boxes foor the given images.

        Arguments:
            images (torch.Tensor): with the batch of images. Shape `(batch size, 3, height, width)`.

        Returns:
            torch.Tensor: The embeddings generated for the images.
                Shape `(batch size, num of embeddings, dim)`.
            torch.Tensor: The bounding boxes for each one of the embeddings.
                Shape `(batch size, num of embeddings, 4)`
        """
        batch_size, _, height, width = images.shape

        if height % 32 != 0:
            raise ValueError('This model only works for images with height multiple of 32.')
        if width % 32 != 0:
            raise ValueError('This model only works for images with width multiple of 32.')

        # Get the height and width of the feature map
        height, width = height / 32, width / 32

        # Generate feature map using the resnet
        features = self.resnet(images)[0]  # (b, f, h, w)

        # Reduce the length of the features by pooling them
        if self.dim != features.shape[1]:
            features = features.view(batch_size, self.dim, -1, height, width)  # (b, d, f-d, h, w)
            if self.pool == 'avg':
                features = features.mean(dim=2)  # (b, d, h, w)
            else:
                features = features.max(dim=2)  # (b, d, h, w)

        # Apply the pooling with kernels and get embeddings
        pooled = []
        for i, pool in enumerate(self.pools):
            kernel = self.kernels[i]
            embeddings = pool(features)  # (b, d, h/k, w/k)
            boxes = self.get_boxes(embeddings, stride=32*kernel, batch_size=batch_size)  # (b, h/k*w/k, 4)
            embeddings = embeddings.view(batch_size, self.dim, -1)  # (b, d, *)
            pooled.append([embeddings, boxes])

        # Transform the feature map to embeddings with shape (batch size, dim, *)
        boxes = self.get_boxes(features, stride=32, batch_size=batch_size)
        embeddings = features.view(batch_size, self.dim, -1)  # (b, d, *)

        # Concatenate all the embeddings
        embeddings = torch.cat([embeddings, *[p[0] for p in pooled]], dim=2)
        boxes = torch.cat([boxes, *[p[1] for p in pooled]], dim=1)

        # Transpose the dimensions to get the embeddings with shape (batch size, num of embeddings, dim)
        embeddings = embeddings.permute(0, 2, 1)

        return embeddings, boxes

    def get_boxes(self, feature_map, stride, batch_size):
        """Get boxes for the given feature map that was got from applying the given stride to the image.

        Arguments:
            feature_map (torch.Tensor): with shape `(batch size, features, height, width)`.
            stride (int): the stride applied to the image to get this feature map.

        Returns:
            torch.Tensor: with the boxes as x1, y1, x2, y2 for top-left corner and bottom-right corner.
                Shape: `(batch size, h * w, 4)` where `h` and `w` are the height and width of the feature map.
        """
        height, width = feature_map.shape[2:]
        boxes = feature_map.new_zeros(height, width, 4)

        for i in range(int(height)):
            for j in range(int(width)):
                boxes[i, j, 0] = stride * i
                boxes[i, j, 1] = stride * j
                boxes[i, j, 2] = stride * (i+1)
                boxes[i, j, 3] = stride * (j+1)

        return boxes.view(-1, 4).unsqueeze(dim=0).repeat((batch_size, 1, 1))
