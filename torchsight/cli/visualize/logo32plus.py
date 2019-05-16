"""Visualize the dataset Logo32plus."""
import click

from torchsight.datasets import Logo32plusDataset


@click.command()
@click.argument('dataset-root')
@click.option('--dataset', default='training', type=click.Choice(['training', 'validation', 'both']))
def logo32plus(dataset_root, dataset):
    """Visualize the images and annotations of the Logo32plus dataset that has its root directory
    at DATASET-ROOT."""
    dataset = Logo32plusDataset(dataset_root, dataset)

    print(len(dataset))

    for i in range(len(dataset)):
        dataset.visualize(i)
