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
        self.transform_no_boxes = Compose(transforms)

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
            dict: with
                image (PIL Image or np.ndarray): The image to transform.
                boxes (np.ndarray or torch.Tensor, optional): The bounding boxes of the image with shape
                    `(num boxes, 4 or 5)`.
                    4 in the case that there aren't labels, and 5 in the case when the bounding boxes have labels.

        Returns:
            torch.Tensor: The transformed image.
            torch.Tensor: The transformed bounding boxes if there is any.
        """
        image, boxes = data['image'], data.get('boxes', None)

        # Transform image to np.ndarray
        if isinstance(image, Image):
            image = np.array(image)

        # Apply the transformation to the image
        if boxes is None:
            return self.transform_no_boxes(image=image)['image']

        # Transform to numpy the boxes if it were tensors
        was_tensor = False
        if torch.is_tensor(boxes):
            was_tensor = True
            boxes = boxes.numpy()

        # Add a dummy label if the boxes does not have one
        had_label = True
        if boxes.shape[1] == 4:
            had_label = False
            num_boxes = boxes.shape[0]
            boxes = np.concatenate([boxes, np.zeros((num_boxes, 1))], axis=1)

        # Apply the transformation
        transformed = self.transform(image=image, bboxes=boxes)
        image, boxes = transformed['image'], transformed.get('bboxes', None)

        # Remove the dummy label
        if not had_label:
            boxes = np.array(boxes)
            boxes = boxes[:, :4]

        # Transform to tensor
        if was_tensor:
            boxes = torch.Tensor(boxes)

        # Return the transformed image and boxes
        return image, boxes
