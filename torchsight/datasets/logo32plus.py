"""The dataset interface to interact with the Logo32plus dataset.

Dataset extracted from:
http://www.ivl.disco.unimib.it/activities/logo-recognition/
"""
import os
import time

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import loadmat


class Logo32plusDataset(torch.utils.data.Dataset):
    """Dataset to get the images and annotations of the Logo32plus dataset.

    Instructions:

    - Download the dataset from:
    http://www.ivl.disco.unimib.it/activities/logo-recognition/
    - Unzip the file in any directory.
    - Provide the path to that directory in the initialization.
    """

    def __init__(self, root, dataset='training', transform=None, annot_file='groundtruth.mat'):
        """Initialize the dataset.

        Arguments:
            root (str): The path where are the unzipped files of te dataset.
            dataset (str, optional): Which dataset to load: 'training', 'validation' or 'both'.
            transform (callable, optional): A callable to transform the image and its bounding boxes
                before return them.
            annot_file (str, optional): The file that contains the annotations for the images.
        """
        self.root = self.validate_root(root)
        self.dataset = self.validate_dataset(dataset)
        self.annotations = self.get_annotations(annot_file)
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

    def get_annotations(self, annot_file):
        """Load and parse the annotations of the images.

        Arguments:
            annot_file (str): The file that contains the annotations.
        """
        annotations = loadmat(os.path.join(self.root, annot_file))['groundtruth'][0]
        result = []
        for annot in annotations:
            image = annot[0][0].replace('\\', '/')
            boxes = self.transform_boxes(annot[1])
            name = annot[2][0]
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
        classes = {annot[2] for annot in self.annotations}
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

        # Append the label to the boxes
        label = self.class_to_label[name]
        n_boxes = boxes.shape[0]
        labels = torch.full((n_boxes,), label)
        boxes = torch.cat([boxes, labels.unsqueeze(dim=1)], dim=1)

        # Load the image
        filepath = os.path.join(self.root, 'images', image)
        image = Image.open(filepath)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes

    def visualize(self, index, initial_time=None):
        """Visualize the annotations for the item in the given index.

        Arguments:
            index (int): The index of the item to visualize.
        """
        initial_time = initial_time if initial_time is not None else time.time()

        image, boxes = self[index]

        if torch.is_tensor(image):
            image = image.numpy().transpose(1, 2, 0)

        # Select n_colors colors from 0 to 1
        n_colors = 20
        colormap = plt.get_cmap('tab20')
        colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]

        _, axes = plt.subplots(1)

        for box in boxes:
            if box.shape[0] == 6:
                x, y, x2, y2, label, prob = box
                prob = ' {:.2f}'.format(prob)
            else:
                x, y, x2, y2, label = box
                prob = ''
            w, h = x2 - x, y2 - y
            color = colors[int(label) % n_colors]
            # Generate and add rectangle to plot
            axes.add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))
            # Generate text if there are any classes
            tag = '{}{}'.format(self.label_to_class[int(label)], prob)
            plt.text(x, y, s=tag, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

        # Print stats
        print('-----\nProcessing time: {}\nBounding boxes:\n{}'.format(time.time() - initial_time, boxes))
        # Show image and plot
        axes.imshow(image)
        plt.show()
