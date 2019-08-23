"""Commands for the few shot experiments with the flickr32 dataset."""
import click


@click.group()
def flickr32():
    """Commands for the few shot experiments with the flickr32 dataset."""


@flickr32.command()
@click.option('-r', '--root', required=True, type=click.Path(exists=True), help='The root directory of the dataset')
@click.option(
    '-f', '--base-file', default='trainvalset.txt', show_default=True,
    help='The base file to use to select the samples'
)
@click.option('-k', default=20, show_default=True)
def generate_dataset(root, base_file, k):
    """Generate the few shot dataset with K samples per class."""
    from torchsight.datasets.flickr32 import Flickr32Dataset

    return Flickr32Dataset.generate_few_shot_dataset(root, base_file, k)