"""Visualize the dataset Logo32plus."""
import random

import click

from torchsight.datasets import Logo32plusDataset


@click.command()
@click.argument('dataset-root')
@click.option('--dataset', default='training', type=click.Choice(['training', 'validation', 'both']))
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
def logo32plus(dataset_root, dataset, no_shuffle):
    """Visualize the images and annotations of the Logo32plus dataset that has its root directory
    at DATASET-ROOT."""
    dataset = Logo32plusDataset(dataset_root, dataset)
    length = len(dataset)
    print('Dataset length: {}'.format(length))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
