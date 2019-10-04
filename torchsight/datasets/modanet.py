"""A module with a PyTorch's dataset class for the Modanet dataset."""
import os

import torch
from PIL import Image

from .mixins import VisualizeMixin


class ModanetDataset(torch.utils.data.Dataset, VisualizeMixin):
    """Modanet dataset to load images with bounding boxes labeling clothes.

    To use the dataset you should have a directory with three directories inside:
    - annotations: with all the annotations for the training images.
    - annotations-valid: with all the annotations for the validation images.
    - images: with all the images of the dataset.
    - images-valid: with all the validation images.

    The annotations files must have a bounding box per line with format:
    `label x y width height` where x and y are the top left corner and label must be 0-indexed.

    And there must be a file `classes.names` with a tab separated file with that maps
    between the label (int) of the class and its name (str).

    The transform must be a callable that accepts a dict with 'image' (PIL.Image) and 'boxes'
    (torch.Tensor with shape (num boxes, 5)) and returns a tuple with the transformed image
    and the transformed boxes with the same shape.
    """

    annotations_dir = 'annotations'
    annotations_valid_dir = 'annotations-valid'
    images_dir = 'images'
    images_valid_dir = 'images-valid'
    classes_names = 'classes.names'

    def __init__(self, root, transform=None, valid=False):
        """Initialize the dataset.

        Arguments:
            root (str): the path to the root directory of the dataset.
            transform (callable): that transform the images and bounding boxes.
            valid (bool): flag to indicate to load the validation dataset.
        """
        self.transform = transform
        self.label_to_class, self.class_to_label = self.get_classes(root)
        self.paths = self.get_paths(root, valid)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.paths)

    def __getitem__(self, i):
        """Get the image with its annotations after applying the transform of the dataset.

        Arguments:
            i (int): the index of the image to load.

        Returns:
            tuple: with the transformed image and boxes.
        """
        image_path, annotations_path = self.paths[i]
        image = Image.open(image_path)
        boxes = self.get_boxes(annotations_path)

        if self.transform is not None:
            image, boxes = self.transform({'image': image, 'boxes': boxes})

        return image, boxes

    def get_paths(self, root, valid):
        """Get a list of tuples with the paths of the images and their bounding boxes.

        Arguments:
            root (str): the path to the root directory of the dataset.
            valid (bool): to indicate to load the validation paths.

        Returns:
            list of tuple of str: with the path for the image and the path for its annotations.
        """
        paths = []
        annotations_dir = os.path.join(root, self.annotations_dir if not valid else self.annotations_valid_dir)
        images_dir = os.path.join(root, self.images_dir if not valid else self.images_valid_dir)

        for name in os.listdir(annotations_dir):
            if name[-3:] == 'txt':
                annotations = os.path.join(annotations_dir, name)
                image = os.path.join(images_dir, name[:-3] + 'jpg')
                paths.append((image, annotations))

        return paths

    def get_classes(self, root):
        """Get the dicts to map between a label (int) to a class' name (str) and between name and label.

        Arguments:
            root (str): the path to the root directory of the dataset.

        Returns:
            dict: to map between label (int) and class' name (str).
            dict: to map between name (str) and label (int).
        """
        with open(os.path.join(root, self.classes_names), 'r') as file:
            label_to_class = {}
            class_to_label = {}

            for line in file.read().split('\n'):
                if line:
                    label, name = line.split()
                    label = int(label) - 1  # The labels are 1-indexed
                    label_to_class[label] = name
                    class_to_label[name] = label

            return label_to_class, class_to_label

    @staticmethod
    def get_boxes(path):
        """Get the bounding boxes of the image that are at the given path.

        Arguments:
            path (str): to the file that contains the annotations with the bounding boxes
                of the image.

        Returns:
            torch.Tensor: The bounding boxes for the given image with shape: `(num of boxes, 5)`
        """
        with open(path, 'r') as file:
            boxes = []

            for line in file.read().split('\n'):
                if line:
                    label, x1, y1, w, h = (int(val) for val in line.split())
                    label = label
                    x2, y2 = x1 + w, y1 + h
                    boxes.append(torch.Tensor([x1, y1, x2, y2, label]))

            return torch.stack(boxes)
