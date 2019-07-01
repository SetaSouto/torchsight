"""A module with transform pipelines to make augmentations."""
import cv2
import numpy as np
import torch
from albumentations import (Compose, GaussianBlur, GaussNoise, LongestMaxSize,
                            Normalize, PadIfNeeded, RandomBrightnessContrast,
                            RandomSizedBBoxSafeCrop, Rotate)
from albumentations.pytorch import ToTensor
from PIL.Image import Image

from torchsight.utils import JsonObject


class AugmentDetection():
    """A custom pipeline to augment the data for a detection task."""

    def __init__(self, params=None, evaluation=False, normalize=True):
        """Initialize the pipeline.

        Arguments:
            params (dict or JsonObject, optional): The params for the pipeline. See get_params().
            evaluation (bool, optional): if True it will not augment the images, only apply the transforms
                to match the sizes.
            normalize (bool, optional): If False it will not apply normalization on evaluation mode.
                Useful to get images without distortion for the human eye.
        """
        self.params = self.get_params().merge(params)

        if evaluation:
            transforms = [
                LongestMaxSize(**self.params.LongestMaxSize),
                PadIfNeeded(**self.params.PadIfNeeded)
            ]
            transforms += [Normalize(), ToTensor()] if normalize else [ToTensor()]
        else:
            transforms = [
                GaussNoise(**self.params.GaussNoise),
                # GaussianBlur(**self.params.GaussianBlur),
                RandomBrightnessContrast(**self.params.RandomBrightnessContrast),
                Rotate(**self.params.Rotate),
                LongestMaxSize(**self.params.LongestMaxSize),
                PadIfNeeded(**self.params.PadIfNeeded),
                RandomSizedBBoxSafeCrop(**self.params.RandomSizedBBoxSafeCrop),
                Normalize(),
                ToTensor()
            ]

        self.transform = Compose(transforms, bbox_params=self.get_box_params())

    @staticmethod
    def get_params():
        """Get the default parameters for the transforms of the pipeline.

        Returns:
            JsonObject: with the params for the transforms.
        """
        return JsonObject({
            'GaussNoise': {
                'var_limit': (10, 50),
                'p': 0.5
            },
            'GaussianBlur': {
                'blur_limit': 0.7,
                'p': 0.5
            },
            'RandomBrightnessContrast': {
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'p': 0.5
            },
            'Rotate': {
                'limit': 45,
                'p': 0.5
            },
            'LongestMaxSize': {
                'max_size': 512
            },
            'PadIfNeeded': {
                'min_height': 512,
                'min_width': 512,
                'border_mode': cv2.BORDER_CONSTANT,
                'value': [0, 0, 0]
            },
            'RandomSizedBBoxSafeCrop': {
                'height': 512,
                'width': 512
            }
        })

    @staticmethod
    def get_box_params():
        """Get the parameters needed for the bounding boxes transforms.

        See: https://github.com/albu/albumentations/blob/master/notebooks/example_bboxes.ipynb

        Returns:
            dict: The params for the bounding boxes transforms.
        """
        return {
            'format': 'pascal_voc',  # like [x_min, y_min, x_max, y_max]
            'min_area': 0,
            'min_visibility': 0
        }

    def __call__(self, data):
        """Apply the transformations.

        Arguments:
            image (PIL Image or np.ndarray): The image to transform.
            boxes (np.ndarray or torch.Tensor): The bounding boxes of the image.
        """
        image, boxes = data['image'], data['boxes']

        if isinstance(image, Image):
            image = np.array(image)

        was_tensor = False
        if torch.is_tensor(boxes):
            was_tensor = True
            boxes = boxes.numpy()

        transformed = self.transform(image=image, bboxes=boxes)

        image, boxes = transformed['image'], transformed['bboxes']

        if was_tensor:
            boxes = torch.Tensor(boxes)

        return image, boxes
