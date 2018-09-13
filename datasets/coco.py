from os import path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from skimage.transform import resize


class CocoDataset(torch.utils.data.Dataset):
    """COCO dataset. It is based on the COCO dataset provided by Joseph Redmon
    at https://pjreddie.com/.
    If you want to get the dataset you can obtain it running this bash script:
    https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

    This class also has a method to get the DataLoader for this dataset. Why to
    create a DataLoader here? Because as the target (tensor with bounding boxes)
    has different shapes every time (depends on the amount of bounding boxes in
    the image) we need a custom collate function to stack our batch (if the batch
    size is over 1).
    For more information please see the get_data_loader() method's documentation.
    """

    def __init__(self, coco_path, image_size=416, train=True, device=None):
        """Given the path to the dataset after running the bash script, load the file
        that indicates the images paths and its labels.

        Args:
            coco_path (str): The path to the coco dataset directory that contains the files '5k.txt'
                and 'trainvalno5k.txt'.
            image_size (int, optional): The size for the width and height of the image.
            train (bool, optional): Boolean to indicate to load the train dataset. If False loads
                the validation dataset.
            device (str, optional): If given move the returned tensor images to that device.
                If no device is present it tries automatically to set the device to use as 'cuda:0'.
        """
        self.image_size = image_size
        self.device = device if device else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        train_path = path.abspath(path.join(coco_path, 'trainvalno5k.txt'))
        valid_path = path.abspath(path.join(coco_path, '5k.txt'))
        file_path = train_path if train else valid_path
        with open(file_path, 'r') as file:
            self.images_paths = [line.strip().replace('\n', '')
                                 for line in file.readlines()]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.images_paths)

    def __getitem__(self, index):
        """Returns the 'index-th' image of the dataset as a torch.Tensor with shape (C, W, H).

        Args:
            index (int): The index of the image to get from the dataset.

        Returns:
            str: Path of the image loaded.
            torch.Tensor: The image as (3, image_size, image_size)
            torch.Tensor: The bounding boxes as (number of bounding boxes, 5).
                See generate_bounding_boxes_tensor().
        """
        image_path = self.images_paths[index % len(self.images_paths)]
        image, real_shape = self.get_image(image_path)
        bounding_boxes = self.generate_bounding_boxes_tensor(
            self.get_bounding_boxes_file_path(image_path), image_shape=real_shape)

        return image, bounding_boxes, image_path

    def get_image(self, image_path):
        """Returns the image normalized [0, 1] as a torch.Tensor with shape (C, W, H).
        Typically C = 3 and the height and width are equal to self.image_size.
        If the image does not fit the image size given, it pads the image to make it a
        square and resize it to match the size.

        Args:
            image_path (str): Path of the image file.

        Returns:
            torch.Tensor: The image as (3, image_size, image_size)
            tuple: The real shape of the image before the editions.
        """
        image = np.array(Image.open(image_path))
        # Handle gray
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=0)
            image = np.repeat(image, 3, axis=0)
            image = np.transpose(image, (1, 2, 0))
        # Get dimensions to define padding
        real_shape = image.shape
        h, w, _ = real_shape
        dim_diff = np.abs(h - w)
        # Upper (or left) and lower (or right) padding
        padding = dim_diff // 2, dim_diff - dim_diff // 2
        padding = (padding, (0, 0), (0, 0)) if h <= w else (
            (0, 0), padding, (0, 0))
        # Pad, normalize and resize
        image = np.pad(image, padding, 'constant',
                       constant_values=127.5) / 255.  # Normalize between [0, 1]
        image = resize(
            image, (self.image_size, self.image_size, 3), mode='reflect')
        # Channels-first
        image = np.transpose(image, (2, 0, 1))
        # As pytorch tensor
        image = torch.from_numpy(image).float()
        # Move to the device
        if self.device:
            image.to(self.device)

        return image, real_shape

    def get_bounding_boxes_file_path(self, image_path):
        """Given the path of the image returns the path of its bounding boxes file."""
        return image_path.replace('images', 'labels').replace('jpg', 'txt')

    def generate_bounding_boxes_tensor(self, file_path, image_shape=None):
        """Read each line of the file to get all the bounding boxes parameters.

        Each line must have 5 values as:
        <object-class> <x> <y> <width> <height>
        Where x, y, width, and height are relative to the image's width and height
        (i.e. between [0, 1]).

        If the image was padded to make it a square of a given dimension the bounding boxes will
        not be correct. To remedy this we have to adjust the bounding boxes to match the actual
        squared shape.
        For this, this method needs the real image shape (the shape of the image before
        it was squared, padded and resized).

        If no image shape is given, ignore the transformations.

        Args:
            file_path (str): The relative or absolute path to the label (or bounding boxes) file.

        Returns:
            torch.Tensor: A Tensor with shape (number of bounding boxes, 5).
        """
        file_path = path.abspath(file_path)
        if path.exists(file_path):
            bounding_boxes = np.loadtxt(file_path).reshape(-1, 5)
            bounding_boxes = torch.from_numpy(bounding_boxes)
            # Move to device
            if self.device:
                bounding_boxes.to(self.device)
            if image_shape:
                # Reformat to match the padded image
                h, w, _ = image_shape
                # The bounding boxes are relative to the image, so we need to know the relative
                # increment of the padding
                if w > h:
                    # We have a vertical padding
                    # The relative height of the image is the proportion between the height and
                    # width.
                    # So if you have an image with height 200 and width 250 the relative height is
                    # 200 / 250 = 0.8
                    relative_height = h / w
                    # The height of the bounding box must scale to this relative height, so if the
                    # height of the bounding box was 1 now it will 0.8, if has 0.5 now it will be
                    # 0.4.
                    bounding_boxes[:, 4] = bounding_boxes[:,
                                                          4] * relative_height
                    # For the y position we need to scale it and move it the correct padding
                    # distance.
                    # For example, if the height and width are 100, 150 we need a padding of
                    # 150 / 100 = 1.5 relative to the height. And then we can add
                    # ((1.5 - 1) * 100) / 2 = 0.25 relative to the height to the top and to the
                    # bottom of the image.
                    # First calculate the padding relative to the height (how many pixels it must
                    # add to match the width length proportionally to the height that it really
                    # has).
                    # It subtracts 1 because it already has all the pixels of the height and
                    # divide by 2 to add the half to the top and the other half to the bottom.
                    padding = ((w / h) - 1) / 2
                    bounding_boxes[:, 2] = (
                        bounding_boxes[:, 2] + padding) * relative_height
                else:
                    relative_width = w / h
                    padding = ((h / w) - 1) / 2
                    bounding_boxes[:, 3] = bounding_boxes[:,
                                                          3] * relative_width
                    bounding_boxes[:, 1] = (
                        bounding_boxes[:, 1] + padding) * relative_width

            return bounding_boxes
        else:
            raise Exception('There is no file at {}'.format(file_path))

    def get_data_loader(self, *args, fill_value=-1.0, **kwargs):
        """Returns a DataLoader for this instance of the dataset.

        The special thing with this method is the custom collate function that it has.
        Also, all the arguments for the DataLoader can be passed through this function.
        For more information about DataLoader class please see:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        Args:
            fill_value (float, optional): Optional value to fill the bounding boxes
                tensors to match all the tensors to the same shape. See collate internal
                function for more information.
        """

        def collate(batch):
            """Collate function to stack the batches into one tensor.

            The DataLoader default collate tries to stack the targets but for to stack the
            tensors pytorch requires that all the tensors has the same shape. This is not
            the case with the bounding boxes tensors because they have dynamic shape as
            (number of bounding boxes, bounding boxes parameters) so we need to get the max number
            of bounding boxes per input in the batch and "fill" the rest of the bounding boxes
            with the given fill_value to stack them.
            So this returns the stack of inputs and the stack of targets "filled".

            Args:
                batch (seq): Sequence of tuples as (input, target, images' paths).
                    Input and target must be torch.Tensor.

            Returns:
                torch.Tensor: Inputs stacked as (batch's size, channels, images' height,
                    images' width).
                torch.Tensor: Bounding boxes stacked as:
                    (batch's size,
                    maximum number of bounding boxes in a single image,
                    bounding boxes parameters)
                tuple: Tuple with the images' paths.
            """
            inputs = torch.stack([item[0] for item in batch])
            images_paths = [item[2] for item in batch]
            targets = [item[1] for item in batch]
            max_shape = max([target.shape[0] for target in targets])
            for i, target in enumerate(targets):
                filling = torch.full((max_shape - target.shape[0], target.shape[1]),
                                     fill_value,
                                     dtype=torch.double)
                targets[i] = torch.cat((target, filling))
            targets = torch.stack(targets)

            return [inputs, targets, images_paths]

        # Body of the method
        return torch.utils.data.DataLoader(dataset=self, collate_fn=collate, *args, **kwargs)

    @staticmethod
    def load_classes_names():
        """Load the classes' names from the coco.names file.

        Returns:
            list: List of strings, list[i] returns the name of the class i.
        """
        classes_path = path.join(path.dirname(__file__), 'coco.names')
        with open(classes_path, 'r') as file:
            return [line.strip().replace('\n', '') for line in file.readlines()]

    def visualize_bounding_boxes(self, image_path=None, image=None, bounding_boxes=None):
        """Show an image and its bounding boxes and classes.

        It can be used giving only the image path or giving already loaded image and bounding boxes.

        Args:
            image_path (str): Path to the image file.
            image (Tensor): A tensor with shape (3, H, W) or (H, W, 3).
            bounding_boxes (Tensor): A tensor with shape (number of bounding boxes, 5).
                Each parameter of a bounding box is interpreted as:
                [object's class, x, y, width, height]
                Where x, y, width, and height are relative to the image's width and height
                (i.e. between [0, 1]).
        """
        if not isinstance(image, torch.Tensor):
            if image_path:
                image, real_image_shape = self.get_image(image_path)
                bounding_boxes = self.generate_bounding_boxes_tensor(
                    self.get_bounding_boxes_file_path(image_path),
                    real_image_shape)
            else:
                raise Exception(
                    "The image must be a torch.Tensor or at least give the image's path")
        if isinstance(image, torch.Tensor) and not isinstance(bounding_boxes, torch.Tensor):
            raise Exception(
                'You have to give the image with its bounding boxes torch.Tensor.')

        # Get the classes to show the names
        classes = self.load_classes_names()

        # Matplotlib colormaps, for more information please visit:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # Is a continuous map of colors, you can get a color by calling it on a number between
        # 0 and 1
        colormap = plt.get_cmap('tab20')
        n_colors = 20
        # Select n_colors colors from 0 to 1
        colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]

        # The image must have a dimension of length 3 to show the image
        if (image.shape[0] != 3) and (image.shape[2] != 3):
            raise Exception('The image must have shape (3, H, W) or (H, W, 3)')

        # Transpose image if it is necessary (numpy interprets the last dimension as the channels,
        # pytorch not)
        image = image if image.shape[2] == 3 else image.permute(1, 2, 0)

        # Generate figure and axes
        _, ax = plt.subplots(1)

        # Generate rectangles
        for i in range(bounding_boxes.shape[0]):
            x, y, w, h = [tensor.item() for tensor in bounding_boxes[i, 1:]]
            # We need the top left corner of the rectangle (or bottom left in values because the
            # y axis is inverted)
            x = x - w / 2
            y = y - h / 2
            # Increase the values to the actual image dimensions, until now this values where
            # between [0, 1]
            image_height, image_width = image.shape[0:2]
            x = x * image_width
            w = w * image_width
            y = y * image_height
            h = h * image_height
            # Select the color for the class
            class_index = int(bounding_boxes[i, 0].item())
            color = colors[class_index % n_colors]
            # Generate and add rectangle to plot
            ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth=2,
                                                      edgecolor=color, facecolor='none'))
            # Generate text if there are any classes
            class_name = classes[class_index]
            plt.text(x, y, s=class_name, color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})
        # Show image and plot
        ax.imshow(image)
        plt.show()
