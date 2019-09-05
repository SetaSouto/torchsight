"""Commands for the few shot experiments with the flickr32 dataset."""
import click


@click.group()
def flickr32():
    """Commands for the few shot experiments with the flickr32 dataset."""


@flickr32.command()
@click.option('-r', '--root', required=True, type=click.Path(exists=True), help='The root directory of the dataset')
@click.option('-bf', '--base-file', default='trainvalset.txt', show_default=True,
              help='The base file to use to select the samples')
@click.option('-k', default=20, show_default=True)
@click.option('-f', '--file-name', show_default=True)
@click.option('--include-no-logo', is_flag=True, help='Include the images without logos (all) in the dataset')
def generate_dataset(root, base_file, k, file_name, include_no_logo):
    """Generate the few shot dataset with K samples per class."""
    from torchsight.datasets.flickr32 import Flickr32Dataset

    return Flickr32Dataset.generate_few_shot_dataset(root, base_file, k, file_name, include_no_logo)


@flickr32.command()
@click.option('-r', '--root', required=True, type=click.Path(exists=True), help='The root directory of the dataset')
@click.option('-f', '--base-file', default='trainvalset.txt', show_default=True)
@click.option('-n', '--num-brands', default=25, show_default=True)
def generate_some_brands_dataset(root, base_file, num_brands):
    """Generate a dataset with only some brands.

    It will write the files `<num_brands>_brands.txt` and `<num_brands>_brands_complement.txt`
    in the root directory of the dataset.
    """
    from torchsight.datasets.flickr32 import Flickr32Dataset

    Flickr32Dataset.generate_some_brands_dataset(root=root, base_file=base_file, num_brands=num_brands)
