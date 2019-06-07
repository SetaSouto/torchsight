"""A dataset interface for the INSTRE dataset.

See official webpage:
http://isia.ict.ac.cn/dataset/instre.html
"""
import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from .mixins import VisualizeMixin


class InstreDataset(Dataset, VisualizeMixin):
    """INSTRE dataset class to get the images with their bounding boxes.

    Instructions:
    - Download the dataset from the original web page.
    - Unzip the dataset to any directory.
    - The root directory is the one that has the 3 directories of the 3 datasets inside.
    """

    def __init__(self, root, name='S2', dataset='training', transform=None):
        """Initialize the dataset.

        Arguments:
            root (str): The path to the root of the directory.
            name (str, optional): The name of the dataset to load.
            dataset (str, optional): The portion of the dataset to load.
            transform (callable, optional): The transformation to apply to the image and bounding boxes.
        """
        self.name = self.validate_name(name)
        self.dataset = self.validate_dataset(dataset)
        self.root = self.validate_root(root)
        self.split = self.get_split()
        self.class_to_label, self.label_to_class = self.generate_labels()
        self.paths = self.generate_paths()
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.paths)

    def __getitem__(self, i):
        """Get the image, bounding boxes and an info dict for the item at the given index.

        Returns:
            any: The transformed image.
            torch.Tensor: The bounding boxes for the image.
                Shape: `(num of boxes, 4)` with x1,y1,x2,y2.
            dict: A dict with additional info.
        """
        image, boxes, name = self.paths[i]

        info = {'image': image, 'class': name}

        image = Image.open(image)
        boxes = self.get_boxes(boxes, name)

        if self.transform is not None:
            image, boxes, info = self.transform((image, boxes, info))

        return image, boxes, info

    def get_boxes(self, file, class_name):
        """Read the file and parse the bounding boxes.

        Return:
            torch.Tensor: The bounding boxes loaded from the file.
                Shape: `(num of boxes, 4)` with x1,y1,x2,y2.
        """
        boxes = []
        label = self.class_to_label[class_name]

        with open(file, 'r') as file:
            for line in file.readlines():
                x, y, w, h = [int(val) for val in line.split()]
                x2, y2 = x + w, y + h
                boxes.append(torch.Tensor([x, y, x2, y2, label]))

        return torch.stack(boxes)

    @staticmethod
    def validate_name(name):
        if name not in ['S1', 'S2', 'M']:
            raise ValueError('You must indicate a valid name. Options: S1, S2 and M. Value: {}'.format(name))

        return 'INSTRE-' + name

    @staticmethod
    def validate_dataset(dataset):
        if dataset not in ['training', 'validation', 'trainval']:
            raise ValueError('Please provide a valid dataset. Options: "training", "validation" and "trainval".')

        return dataset

    def validate_root(self, root):
        if not os.path.exists(root):
            raise ValueError('"{}" does not exists.'.format(root))

        if self.name not in os.listdir(root):
            raise ValueError('The dataset "{}" is not the root directory. '
                             'Are you sure that is the correct directory?'.format(self.name))

        return os.path.join(root, self.name)

    def get_split(self):
        """Get the split file or generate a new one.

        Returns:
            dict: A dict with all the class names with a 'training' and a 'validation'
                list of of images paths.
        """
        file = os.path.join(self.root, 'split.json')

        if not os.path.exists(file):
            print('There is no split for the dataset, generating a new random one.')
            return self.split()

        with open(file, 'r') as file:
            return json.loads(file.read())

    def generate_labels(self):
        """Generate the dicts to map the labels with the classes.

        Returns:
            dict: A dict that maps the class (str) -> label (int).
            dict: A dict that maps the label (int) -> class (str).
        """
        classes = [name.replace('_', ' ') for name in self.split.keys()]
        classes.sort()

        return (
            {name: i for i, name in enumerate(classes)},
            {i: name for i, name in enumerate(classes)}
        )

    def generate_paths(self):
        """Generate tuples with (image file, boxes file, class name) for each image in the dataset.

        Returns:
            list of tuples: A list with tuples like (image file, boxes file, class name).
        """
        paths = []

        for name in self.split.keys():
            images = []

            if self.dataset in ['training', 'trainval']:
                images += self.split[name]['training']
            if self.dataset in ['validation', 'trainval']:
                images += self.split[name]['validation']

            for image in images:
                boxes = image[:-3] + 'txt'
                paths.append((image, boxes, name.replace('_', ' ')))

        return paths

    def make_split(self, proportion=0.8):
        """Make a random split of the given dataset.

        It will look for all the images inside each class and will generate a JSON file with
        the training images and the validation images for each class.

        Arguments:
            proportion (float, optional): The proportion of training images to select from the total
                number of images.
        """
        classes = [directory for directory in os.listdir(self.root)
                   if os.path.isdir(os.path.join(self.root, directory))]

        split = {}

        for class_name in classes:
            images = [os.path.join(self.root, class_name, file)
                      for file in os.listdir(os.path.join(self.root, class_name))
                      if file[-4:] == '.jpg']

            training = random.sample(images, int(len(images) * proportion))
            validation = list(set(images) - set(training))

            split[class_name] = {
                'training': training,
                'validation': validation
            }

        split_file = os.path.join(self.root, 'split.json')
        with open(split_file, 'w') as file:
            file.write(json.dumps(split))

        self.split = split
        self.print_stats()

        return split

    def print_stats(self):
        """Print the distribution of the classes in the given split json file."""
        classes = [class_name for class_name in self.split]

        print('Classes: {}'.format(len(classes)))

        for name in classes:
            print('{} {} {}'.format(
                name.ljust(30, '.'),
                str(len(self.split[name]['training'])).rjust(3),
                str(len(self.split[name]['validation'])).rjust(3)))
