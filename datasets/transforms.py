"""Useful transforms for the images for any dataset.

The recomendation is to compose the transforms in the order that are written:
Resize(), ToTensor(), Normalize().
"""

import numpy as np
import torch
from skimage.transform import resize

from torchvision.transforms.functional import normalize, to_tensor


class Resize():
    """Resize an image and pad it to fit the square.

    It takes the size of the required square and make the largest side of the image take this length.

    It works with a tuple where it first value is a ndarray image and the second value
    are the bounding boxes.
    """

    def __init__(self, size=800):
        self.size = size

    def __call__(self, data):
        """Resize the image and scale the bounding boxes.

        Args:
            data (tuple): A tuple with a PIL image and the bounding boxes as numpy arrays.
        """
        image, bounding_boxes = data
        height, width, channels = image.shape

        if height > width:
            scale = self.size / height
            new_width = int(round(width * scale))
            new_height = self.size
        else:
            scale = self.size / width
            new_width = self.size
            new_height = int(round(height * scale))

        image = resize(image, (new_height, new_width))
        height, width, channels = image.shape

        square = np.zeros((self.size, self.size, channels))
        square[:height, :width, :] = image

        bounding_boxes[:, :4] *= scale

        return square, bounding_boxes

class ToTensor():
    """Transform a tuple with a PIL image or ndarray and bounding boxes to tensors.

    See: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L38
    """

    def __call__(self, data):
        """Transforms the image and bounding boxes to tensors.

        Args:
            data (tuple): A tuple with a PIL image and the bounding boxes as numpy arrays.
        """
        return to_tensor(data[0]), torch.from_numpy(data[1])


class Normalize():
    """Normalize an image by a mean and standard deviation.

    See: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L157

    It works with a tuple and it assumes that the first element is the image as a tensor.
    """

    def __init__(self, mean=None, std=None):
        """Initialize the normalizer with the given mean and std.

        If there is no mean or std it assigns them automatically.
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.mean = mean
        self.std = std

    def __call__(self, data):
        """Normalize the first element of the tuple assuming that is an image.

        Args:
            data (tuple): A tuple where it first element is an image as a tensor.
        """
        image, *rest = data
        return (normalize(image, self.mean, self.std), *rest)
