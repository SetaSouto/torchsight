"""Datasets for the InstanceRetrievers."""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImagesDataset(Dataset):
    """A dataset to load the images."""

    def __init__(self, root, transform=None, extensions=None):
        """Initialize the dataset.

        Arguments:
            root (str): The path to the root directory that contains the images
                to generate the database.
            transform (callable, optional): The transform to apply to the image.
            extensions (list of str): If given it will load only files with the
                given extensions.
        """
        self.root = root
        self.transform = transform
        self.extensions = extensions
        if extensions is not None:
            self.extensions = extensions if isinstance(extensions, (list, tuple)) else [extensions]
        self.images = self.get_images_paths()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        """Load an image.

        Arguments:
            i (int): The index of the image to load.

        Returns:
            image: The image loaded and transformed.
        """
        path = self.images[i]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, path

    def get_images_paths(self):
        """Get all the paths of the images that are in the given directory
        and its subdirectories.
        """
        if not os.path.exists(self.root):
            raise ValueError('The directory "{}" does not exists.'.format(self.root))

        images = []
        for dirpath, _, files in os.walk(self.root):
            images += [os.path.join(dirpath, file) for file in files if self.is_valid(file)]

        return images

    def is_valid(self, file):
        """Check if the file has a correct extension. If we don't have extensions to check
        it always returns True.

        Arguments:
            file (str): The file's name to check.
        """
        if self.extensions is None:
            return True

        return any((file.endswith(ext) for ext in self.extensions))

    def get_dataloader(self, batch_size, num_workers):
        """Get the dataloader for this dataset.

        Returns:
            DataLoader: the dataloader using the given parameters.
        """
        def collate(items):
            images = [item[0] for item in items]
            paths = [item[1] for item in items]

            if torch.is_tensor(images[0]):
                max_width = max([image.shape[2] for image in images])
                max_height = max([image.shape[1] for image in images])

                def pad_image(image):
                    aux = torch.zeros((image.shape[0], max_height, max_width))
                    aux[:, :image.shape[1], :image.shape[2]] = image
                    return aux

                images = torch.stack([pad_image(image) for image in images], dim=0)

            return images, paths

        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
