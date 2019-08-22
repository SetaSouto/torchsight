"""A dataset interface for the Flickr32 dataset.

Original web page for more information:
http://www.multimedia-computing.de/flickrlogos/
"""
import os

import torch
from PIL import Image

from torchsight.utils import describe_boxes

from .mixins import VisualizeMixin


class Flickr32Dataset(torch.utils.data.Dataset, VisualizeMixin):
    """Dataset to get the images and annotations of the Flickr32 dataset.

    Download the dataset:
    - Request the dataset zip file in the original web page:
      http://www.multimedia-computing.de/flickrlogos/
    - Unzip the dataset in any directory.
    - Provide the path to the root* directory of the dataset in the initialization.

    *: The root directory is the one that contains the 'classes' and 'scripts' directories
    and the `.txt` files with the split of the data (training, validation and test sets).
    """

    def __init__(self, root, dataset='training', transform=None, brands=None, only_boxes=False):
        """Initialize the dataset.

        Arguments:
            root (str): The path to the root directory that contains the data.
            dataset (str, optional): The dataset that you want to load.
                Options available: 'training', 'validation', 'test', 'trainval'.
            transform (callable, optional): A callable to transform the images and
                bounding boxes.
            brands (list, optional): A list with the brands to load. If None is provided it will load
                all the classes.
            only_boxes (bool, optional): If True, it will load images that only contains bounding boxes.
                This is an option because in the validation and test set there are images without logos,
                but for training we probably don't want to train with that images.
        """
        self.root = self.validate_root(root)
        self.dataset = self.validate_dataset(dataset)
        self.transform = transform
        self.brands = brands
        self.only_boxes = only_boxes
        self.paths = self.get_paths()
        self.label_to_class, self.class_to_label = self.generate_labels()

        if self.brands is None:
            self.brands = list(self.class_to_label.keys())

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.paths)

    def __getitem__(self, i):
        """Get the image and bounding boxes at the given index.

        Arguments:
            i (int): The index of the element that you want to get.

        Returns:
            any: The transformed image.
            torch.Tensor: The bounding boxes with x1, y1, x2, y2, label.
                Shape: `(num of boxes, 5)`
            dict: A dict with more info about the item like the brand name and the
                path to the image.
        """
        brand, image, boxes = self.paths[i]

        info = {'brand': brand, 'image': image}

        image = Image.open(image)
        boxes = self.get_boxes(boxes, brand)

        if self.transform is not None:
            image, boxes = self.transform({'image': image, 'boxes': boxes})

        return image, boxes, info

    ################################
    ###        VALIDATION        ###
    ################################

    @staticmethod
    def validate_root(path):
        """Validate that the given path exists and return it.

        Arguments:
            path (str): The path to validate that exists.

        Returns:
            str: The given path if its valid.

        Raises:
            ValueError: if the given path does not exists.
        """
        if not os.path.exists(path):
            raise ValueError('"{}" does not exists.'.format(path))

        return path

    @staticmethod
    def validate_dataset(name):
        """Check that the given name of the dataset is a correct one.

        Arguments:
            name (str): The name of the dataset to check.

        Returns:
            str: The name if its valid.

        Raises:
            ValueError: if the given name is not a valid one.
        """
        if name not in ['training', 'validation', 'test', 'trainval']:
            raise ValueError('"{}" is not a valid dataset name.'.format(name))

        return name

    ############################
    ###       GETTERS        ###
    ############################

    def get_paths(self):
        """Load the absolute paths to the files that are in the dataset.

        Returns:
            list of tuples of str: Each tuple containing the class' name, the
                path to the image and the path to its bounding boxes.
                Example:
                ('google',
                 '/datasets/flickr32/classes/jpg/google/2240784196.jpg',
                 '/datasets/flickr32/classes/masks/google/2240784196.jpg.bboxes.txt')
                If the image has no brand it set the brand as 'no-logo'.
        """
        if self.dataset == 'training':
            file = 'trainset.txt'
        if self.dataset == 'validation':
            file = 'valset.txt'
        if self.dataset == 'test':
            file = 'testset.txt'
        if self.dataset == 'trainval':
            file = 'trainvalset.txt'

        file = os.path.join(self.root, file)

        with open(file, 'r') as file:
            tuples = []

            for line in file.readlines():
                brand, image = line.split(',')

                if self.brands is not None and brand not in self.brands:
                    continue

                image = image.replace('\n', '')
                boxes = os.path.join(
                    self.root, 'classes/masks/{}/{}.bboxes.txt'.format(brand if brand != 'HP' else 'hp', image))
                image = os.path.join(self.root, 'classes/jpg/{}/{}'.format(brand, image))

                if not os.path.exists(boxes) and self.only_boxes:
                    continue

                tuples.append((brand, image, boxes))

            return tuples

    def generate_labels(self):
        """Generate the labels for the classes.

        Returns:
            dict: A dict with the label (int) -> brand (str) map.
            dict: A dict with the brand (str) -> label (int) map.
        """
        brands = list({brand for brand, *_ in self.paths if brand != 'no-logo'})
        brands.sort()

        label_to_class = {i: brand for i, brand in enumerate(brands)}
        class_to_label = {brand: i for i, brand in enumerate(brands)}

        label_to_class[-1] = 'no-logo'
        class_to_label['no-logo'] = -1

        return label_to_class, class_to_label

    def get_boxes(self, file, brand):
        """Get the boxes from the given file.

        Arguments:
            file (str): The path to the file that contains the annotations.
            brand (str): The name of the brand for the boxes.

        Returns:
            torch.Tensor: The bounding boxes for the given image.
                Shape: `(num of boxes, 5)`
        """
        label = self.class_to_label[brand]

        try:
            with open(file, 'r') as file:
                boxes = []

                for line in file.readlines()[1:]:  # The first line contains "x y width height"
                    x, y, w, h = (int(val) for val in line.split())
                    x1, y1 = x - 1, y - 1
                    x2, y2 = x1 + w, y1 + h
                    boxes.append(torch.Tensor([x1, y1, x2, y2, label]))

                return torch.stack(boxes)
        except FileNotFoundError:
            return torch.Tensor([])

    ##########################
    ###       STATS        ###
    ##########################

    def describe_boxes(self):
        """Describe the boxes of the dataset.

        See torchsight.utils.describe_boxes docs for more information.
        """
        if self.transform is not None:
            boxes = []
            for i, (_, bxs, *_) in enumerate(self):
                print('Loading boxes ... ({}/{})'.format(i + 1, len(self)))
                boxes.append(bxs)
        else:
            boxes = [self.get_boxes(boxes, brand) for brand, _, boxes in self.paths]

        return describe_boxes(torch.cat(boxes, dim=0))

    #############################################
    ###          DATASETS GENERATORS          ###
    #############################################

    @staticmethod
    def generate_few_shot_dataset(root, base_file='trainvalset.txt', k=20):
        """Generate a new dataset based on the given one that takes
        only `k` samples per class.

        It will generate in the root directory a file named `few_shot_k.txt`
        where `k` will be replaced by the given one.

        Arguments:
            root (str): path of the directory that contains the dataset and the base file.
            base_file (str, optional): name of the file that will be used to generate the
                new dataset.
            k (int, optional): number of samples to use per class.
        """
        # import here the random package because is never used in the other methods and
        # this method is rarely used
        import random

        images = {}  # images keyed by brand

        # Get all the images per brand from the file
        with open(os.path.join(root, base_file), 'r') as file:
            for line in file.readlines():
                brand, image = line.split(',')

                if brand == 'no-logo':
                    continue

                if brand not in images:
                    images[brand] = []

                images[brand].append(image)

        # Select randomly only k images
        for brand in images:
            images[brand] = random.sample(images[brand], k)

        # Save the new file
        file_name = 'few_shot_{}.txt'.format(k)
        with open(os.path.join(root, file_name), 'w') as file:
            for brand in images:
                for image in images[brand]:
                    file.write('{},{}'.format(brand, image))
