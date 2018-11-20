"""Coco dataset module.

It contains a more robust and flexible version of the Coco dataset.
Please prefere to use this dataset than the old one.
"""
import os
import time

import matplotlib
import numpy as np
import skimage
import skimage.io
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """Coco dataset class.

    Heavily inspired by the code at:
    https://github.com/SetaSouto/pytorch-retinanet/blob/master/dataloader.py
    """

    def __init__(self, root, dataset, classesNames=(), transform=None, stats=True):
        """Initialize the dataset.

        Initialize the coco api under the 'coco' attribute.

        Also, loads the classes and labels.

        It sets the 'classes' attribute of the class that contains a dict with four dicts:
        - 'ids': Keep track of the classes' ids given its label (int) (label -> id).
        - 'labels': Keep track of the label (int) given the id of the class (id -> label).
        - 'names': Keep track of the name of the class given its label (label -> name).
        - 'length': Keep track of how many images are for the given label (label -> int).

        Why labels and ids? Because ids are given by the coco api, are static, but if we filter the classes
        we want to reorder the labels from 0 to N if we load N classes.
        For example, the 2017 dataset contains classes with ids from 1 to 90, but we like to keep labels
        starting from 0 (zero indexed). If we want only to load the classes 'person' (id: 1), 'bus' (id: 6)
        and 'stop sign' (id: 13) we don't want to work with labels 1, 6 and 13, we want to work with labels
        0, 1, 2, because the loaded only 3 images.

        You can filter the images and classes to be loaded by using classesNames argument.

        Why we load the bounding boxes already? Because the COCO api already does that, so instead of keeping
        the coco instance we format the bounding boxes in the initialization and keep track of the path
        of each image. The images are loaded in running time.
        This way we use the api in the initialization only and free some memory.

        Args:
            root (str): COCO root directory.
                Inside this directory we must find the 'annotations' and 'images' folder for example.
            dataset (str): The name of the set to be loaded. Is the name of the directory that contains
                the images and is in the name of the file that contains the annotations.
                Example: 'train2017' will trigger the loading of the images at coco/images/train2017
                and the annotations from the file 'instances_train2017.json'.
            classesNames (tuple): Tuple of strings with the name of the classes to load. Only load images with those
                classes' names.
            transform (torchvision.transforms.Compose, optional): A list with transforms to apply to each image.
            stats (Boolean): Print the stats of the classes. Indicates how many images are per category.
                Keep in mind that the sum of all the images per category may not be the same to the total number
                of images, this is because some images could contain more than one category (very common).
        """
        self.transform = transform

        # Initialize the COCO api
        print('--- COCO API ---')
        coco = COCO(os.path.join(root, 'annotations', 'instances_{}.json'.format(dataset)))
        print('----------------')

        # Load classes and set classes dict
        print('Loading classes and setting labels, names and lengths ...')
        self.classes = {'ids': {}, 'names': {}, 'labels': {}, 'length': {}}
        for label, category in enumerate(coco.loadCats(coco.getCatIds(catNms=classesNames))):
            self.classes['ids'][label] = category['id']
            self.classes['labels'][category['id']] = label
            self.classes['names'][label] = category['name']

        # Get filtered images ids
        print('Loading images info ...')
        images_ids = set()
        for category_id in self.classes['ids'].values():
            category_images = set(coco.catToImgs[category_id])
            category_label = self.classes['labels'][category_id]
            self.classes['length'][category_label] = len(category_images)
            images_ids |= category_images

        # Init images array that contains the path to the image and the annotations
        print('Setting bounding boxes ...')
        self.images = []
        for image_info in coco.loadImgs(images_ids):
            bounding_boxes = np.zeros((0, 5))

            for annotation in coco.imgToAnns[image_info['id']]:
                try:
                    label = self.classes['labels'][annotation['category_id']]
                    bounding_boxes = np.append(bounding_boxes, np.array([[*annotation['bbox'], label]]), axis=0)
                except KeyError:
                    # The image has a bounding box not relevant for us (given the classesNames)
                    continue

            self.images.append((
                os.path.join(root, 'images', dataset, image_info['file_name']),  # Image's path
                bounding_boxes  # Bounding boxes with shape (N, 5)
            ))

        # Print stats
        if stats:
            print('Images per class:')
            stats = [(label,
                      self.classes['names'][label],
                      self.classes['length'][label])
                     for label in self.classes['labels'].values()]
            stats = sorted(stats, key=lambda x: x[2], reverse=True)
            for label, name, length in stats:
                print('{}: {}{}'.format(str(label).ljust(2), name.ljust(15), length))

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """Get an item from the dataset given its index.

        Load the image and bounding boxes for that image and return a tuple with
        image, bounding boxes. Both are ndarrays.

        The bounding boxes is a numpy array with shape (N, 5) where N is the number
        of bounding boxes in the image. Why 5? Because they are x1, y1 for the top
        left corner of the bounding box, x2, y2 for the bottom right corner of the
        bounding box and the last one is the label of the class.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            ndarray: The transformed image if there is any transformation.
            ndarray: The bounding boxes of the image.
        """
        path, bounding_boxes = self.images[index]

        image = skimage.io.imread(path)
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
        image = image.astype(np.float32) / 255.0

        # Transform from [x,y,w,h] to [x1,y1,x2,y2]
        bounding_boxes[:, 2] = bounding_boxes[:, 0] + bounding_boxes[:, 2]
        bounding_boxes[:, 3] = bounding_boxes[:, 1] + bounding_boxes[:, 3]

        if self.transform:
            image, bounding_boxes = self.transform((image, bounding_boxes))

        return image, bounding_boxes

    def visualize(self, index, bounding_boxes=True):
        """Visualize an image with its bounding boxes.

        Args:
            index (int): The index of the image in the dataset to be loaded.
        """
        initial_time = time.time()

        image, bounding_boxes = self.__getitem__(index)

        if torch.is_tensor(image):
            image = image.numpy().transpose(1, 2, 0)

        # Matplotlib colormaps, for more information please visit:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # Is a continuous map of colors, you can get a color by calling it on a number between
        # 0 and 1
        colormap = matplotlib.pyplot.get_cmap('tab20')
        n_colors = 20
        # Select n_colors colors from 0 to 1
        colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]

        # Generate figure and axes
        _, axes = matplotlib.pyplot.subplots(1)

        # Generate rectangles
        for i in range(bounding_boxes.shape[0]):
            # We need the top left corner of the rectangle and its width and height
            x, y, x2, y2, label = bounding_boxes[i]
            w, h = x2 - x, y2 - y
            color = colors[int(label) % n_colors]
            # Generate and add rectangle to plot
            axes.add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))
            # Generate text if there are any classes
            class_name = self.classes['names'][int(label)]
            matplotlib.pyplot.text(x, y, s=class_name, color='white',
                                   verticalalignment='top', bbox={'color': color, 'pad': 0})
        # Print stats
        print('----------\nProcessing time: {}\nBounding boxes:\n{}'.format(time.time() - initial_time, bounding_boxes))
        # Show image and plot
        axes.imshow(image)
        matplotlib.pyplot.show()
