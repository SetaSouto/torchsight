"""Functional module to get dataloaders."""
import torch
from torch.utils.data import DataLoader


def get_dataloaders(hyperparameters, train_dataset, valid_dataset=None):
    """Initialize and get the dataloaders for the given datasets.

    This is function is here to avoid duplication of code, as almost all the datasets
    returns tuples of (images, batches, *_) we generate dataloaders for that datasets
    with a single function.

    Arguments:
        hyperparameters (dict): A dict with the keyword arguments for the DataLoader.
        train_dataset (torch.utils.data.Dataset): The dataset for training.
        valid_dataset (torch.utils.data.Dataset, optional): An optional dataset for validation.

    Returns:
        torch.utils.data.Dataloaders: The dataloader for the training dataset.
        torch.utils.data.Dataloaders: The dataloader for the validation dataset.
    """
    def collate(data):
        """Custom collate function to join the different images with its different annotations.

        Why is this important?
        Because as each image could contain different amounts of annotated objects the tensor
        for the batch could not be created, so we need to "fill" the annotations tensors with -1
        to give them the same shapes and stack them.
        Why -1?
        Because the FocalLoss could interpret that label and ingore it for the loss.

        Also it pads the images so all has the same size.

        Arguments:
            data (sequence): Sequence of tuples as (image, annotations, *_).

        Returns:
            torch.Tensor: The images.
                Shape:
                    (batch size, channels, height, width)
            torch.Tensor: The annotations.
                Shape:
                    (batch size, biggest amount of annotations, 5)
        """
        images = [image for image, *_ in data]
        max_width = max([image.shape[-1] for image in images])
        max_height = max([image.shape[-2] for image in images])

        def pad_image(image):
            aux = torch.zeros((image.shape[0], max_height, max_width))
            aux[:, :image.shape[1], :image.shape[2]] = image
            return aux

        images = torch.stack([pad_image(image) for image, *_ in data], dim=0)

        max_annotations = max([annotations.shape[0] for _, annotations, *_ in data])

        def fill_annotations(annotations):
            aux = torch.ones((max_annotations, 5))
            aux *= -1
            aux[:annotations.shape[0], :] = annotations
            return aux

        annotations = torch.stack([fill_annotations(a) for _, a, *_ in data], dim=0)
        return images, annotations

    hyperparameters = {**hyperparameters, 'collate_fn': collate}
    train_dataloader = DataLoader(**hyperparameters, dataset=train_dataset)

    if valid_dataset is not None:
        return (train_dataloader,
                DataLoader(**hyperparameters, dataset=valid_dataset))

    return train_dataloader
