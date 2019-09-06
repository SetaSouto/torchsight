"""The dataset interface to interact with the Logo32plus dataset.

Dataset extracted from:
http://www.ivl.disco.unimib.it/activities/logo-recognition/
"""
import json
import math
import os
import random

import torch
from PIL import Image
from scipy.io import loadmat

from .mixins import VisualizeMixin


class Logo32plusDataset(torch.utils.data.Dataset, VisualizeMixin):
    """Dataset to get the images and annotations of the Logo32plus dataset.

    Instructions:

    - Download the dataset from:
    http://www.ivl.disco.unimib.it/activities/logo-recognition/
    - Unzip the file in any directory.
    - Provide the path to that directory in the initialization.
    """

    def __init__(self, root, dataset='training', transform=None, annot_file='groundtruth.mat',
                 classes=None, split_file='train_valid.json'):
        """Initialize the dataset.

        Arguments:
            root (str): The path where are the unzipped files of te dataset.
            dataset (str, optional): Which dataset to load: 'training', 'validation' or 'both'.
            transform (callable, optional): A callable to transform the image and its bounding boxes
                before return them.
            annot_file (str, optional): The file that contains the annotations for the images.
            classes (list of str, optional): Only load this classes (identified by its name).
            split_file (str, optional): The file that contains the split between training and validation
                sets.
        """
        self.root = self.validate_root(root)
        self.dataset = self.validate_dataset(dataset)
        self.annot_file = annot_file
        self.classes = classes
        self.split = self.get_split(split_file)
        self.annotations = self.get_annotations()
        self.label_to_class, self.class_to_label = self.generate_classes()
        self.transform = transform

    @staticmethod
    def validate_root(root):
        """Validate that the root path already exists.

        Arguments:
            root (str): The path to validate.

        Returns:
            str: The path if it's correct.

        Raises:
            ValueError: When the path does not exists.
        """
        if not os.path.exists(root):
            raise ValueError('There is no directory with path: {}'.format(root))

        return root

    @staticmethod
    def validate_dataset(dataset):
        """Validate that the dataset is in ['training', 'validation', 'both'].

        Arguments:
            dataset (str): The string to validate.

        Returns:
            str: The dataset if it's valid.

        Raises:
            ValueError: If the given dataset is not a valid one.
        """
        if dataset not in ['training', 'validation', 'both']:
            raise ValueError('The dataset must be "training", "validation" or "both", not "{}"'.format(dataset))

        return dataset

    def get_split(self, split_file):
        """Get the JSON with the split file or generate a new one.

        Arguments:
            split_file (str): The name of the file that contains the json with the split.
        """
        filepath = os.path.join(self.root, split_file)

        if not os.path.exists(filepath):
            self.generate_split(annotations=self.get_annotations(), split_file=split_file)

        with open(filepath, 'r') as file:
            return json.loads(file.read())

    def get_annotations(self):
        """Load and parse the annotations of the images.

        Returns:
            list of tuples: like (image: str, boxes: tensor, name: str)
        """
        annotations = loadmat(os.path.join(self.root, self.annot_file))['groundtruth'][0]
        result = []
        for annot in annotations:
            name = annot[2][0]
            if self.classes is not None and name not in self.classes:
                continue

            image = annot[0][0].replace('\\', '/')
            if self.dataset != 'both' and getattr(self, 'split', None) is not None and image not in self.split[self.dataset]:
                continue

            boxes = self.transform_boxes(annot[1])
            result.append((image, boxes, name))

        return result

    def transform_boxes(self, boxes):
        """Transform the boxes with x,y,w,h 1-indexed to x1,y1,x2,y2 0-indexed.

        Arguments:
            boxes (list of list of int): A list with the annotations in format x,y,w,h 1-indexed.
        """
        boxes = torch.Tensor(boxes.astype('int32'))
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = x - 1, y - 1  # 0-indexed
        x2, y2 = x1 + w, y1 + h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        return boxes

    def generate_classes(self):
        """Generate the map dicts to assign a 0-indexed label to each one of the classes and viceversa."""
        classes = list({annot[2] for annot in self.annotations})
        classes.sort()
        label_to_class = {i: c for i, c in enumerate(classes)}
        class_to_label = {c: i for i, c in enumerate(classes)}

        return label_to_class, class_to_label

    def __len__(self):
        """Get the number of images in this dataset."""
        return len(self.annotations)

    def __getitem__(self, index):
        """Get an item from the dataset.

        Arguments:
            index (int): The index of the item that you want to get.

        Returns:
            tuple: A tuple with the image and the bounding boxes.
                The image is a PIL image or the result of the callable transform.
                The bounding boxes are a torch tensor with shape (num annot, 5),
                because an image could have more than one annotation and the 5 values are
                x1,y1,x2,y2 and the label.
        """
        image, boxes, name = self.annotations[index]

        info = {'brand': name, 'file': image}

        # Append the label to the boxes
        label = self.class_to_label[name]
        n_boxes = boxes.shape[0]
        labels = torch.full((n_boxes,), label)
        boxes = torch.cat([boxes, labels.unsqueeze(dim=1)], dim=1)

        # Load the image
        filepath = os.path.join(self.root, 'images', image)
        image = Image.open(filepath)

        if self.transform:
            image, boxes = self.transform({'image': image, 'boxes': boxes})

        return image, boxes, info

    def generate_split(self, annotations, proportion=0.8, split_file='train_valid.json'):
        """Create the validation and training datasets with the given proportion.

        The proportion is used in each class. For example, with a proportion of 0.8 and a class with
        20 elements, this method creates a training dataset with 16 of those 20 images.

        Arguments:
            proportion (float): A float between [0, 1] that is the amount of training samples extracted
                from the total samples in each class.
        """
        brands = {}
        training = {}
        validation = {}

        for image, _, brand in annotations:
            if brand not in brands:
                brands[brand] = set()
                training[brand] = set()
                validation[brand] = set()

            brands[brand].add(image)

        result = {'training': [], 'validation': []}

        for brand, images in brands.items():
            n_train = math.ceil(len(images) * proportion)
            train = set(random.sample(images, n_train))
            valid = images - train

            result['training'] += list(train)
            result['validation'] += list(valid)

        with open(os.path.join(self.root, split_file), 'w') as file:
            file.write(json.dumps(result))
