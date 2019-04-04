"""Useful transforms for the images for any dataset for object detection.

The recomendation is to compose the transforms in the order that are written:
Resize(), ToTensor(), Normalize().
"""

import numpy as np
import torch
from skimage.transform import resize

from torchvision.transforms.functional import normalize, to_tensor


class Resize():
    """Resize an image to fit between the min_side and max_side.

    It tries to match the smallest side of the image to the min_side attribute of this transform
    and if the biggest side of the image after the transformation will be over the max_size attribute
    it instead resize the image to match the biggest side to the max_size attribute.

    Also, it tries to keep a multiple of the stride attribute on each of the sides to match design
    better the feature map.
    """

    def __init__(self, min_side=640, max_side=1024, stride=128):
        self.min_side = min_side
        self.max_side = max_side
        self.stride = stride

    def __call__(self, data):
        """Resize the image and scale the bounding boxes.

        Args:
            data (tuple): A tuple with a PIL image and the bounding boxes as numpy arrays.
        """
        image, bounding_boxes = data
        height, width, channels = image.shape

        smallest_side = height if height < width else width
        biggest_side = height if height > width else width

        scale = self.min_side / smallest_side
        scale = self.max_side / biggest_side if scale * biggest_side > self.max_side else scale

        new_width = round(width * scale)
        new_height = round(height * scale)

        padding_width = self.stride - (new_width % self.stride)
        padding_width = 0 if padding_width == 32 else padding_width
        padding_height = self.stride - (new_height % self.stride)
        padding_height = 0 if padding_height == 32 else padding_height

        image = resize(image, (new_height, new_width), mode='constant', anti_aliasing=True)
        height, width, channels = image.shape

        final = np.zeros((new_height + padding_height, new_width + padding_width, channels))
        final[:height, :width, :] = image

        bounding_boxes[:, :4] *= scale

        return final, bounding_boxes


class ToTensor():
    """Transform a tuple with a PIL image or ndarray and bounding boxes to tensors.

    See: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L38
    """

    def __call__(self, data):
        """Transforms the image and bounding boxes to tensors.

        Arguments:
            data (tuple): A tuple with a PIL image and the bounding boxes as numpy arrays.

        Returns:
            torch.Tensor: The image.
            torch.Tensor: The annotations.
        """
        return to_tensor(data[0]), torch.from_numpy(data[1])


class Normalize():
    """Normalize an image by a mean and standard deviation.

    See: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L157

    It works with a tuple and it assumes that the first element is the image as a tensor.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Initialize the normalizer with the given mean and std.

        Arguments:
            mean (sequence): Sequence of floats that contains the mean to which normalize each channel.
            std (sequence): The standard deviation for each of the channels.
        """

        self.mean = mean
        self.std = std

    def __call__(self, data):
        """Normalize the first element of the tuple assuming that is an image.

        Arguments:
            data (tuple): A tuple where it first element is an image as a tensor.

        Returns:
            torch.Tensor: The image normalized.
        """
        image, *rest = data
        image = image.type(torch.float)
        return (normalize(image, self.mean, self.std), *rest)
