"""Commands to see stats about the bounding boxes of a dataset."""
import click


@click.command()
@click.option('-d', '--dataset', required=True, type=click.Choice(['flickr32']))
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True))
@click.option('--resize', is_flag=True)
@click.option('--min-side', default=384, show_default=True)
@click.option('--max-side', default=512, show_default=True)
@click.option('--stride', default=128, show_default=True)
def describe_boxes(dataset, dataset_root, resize, min_side, max_side, stride):
    """Describe the shapes of the bounding boxes.

    It computes the min, max, mean, median of the height, width and area of the bounding boxes.
    """
    from torchsight.datasets import Flickr32Dataset

    transform = None

    if resize:
        from torchsight.transforms.detection import Resize

        transform = Resize(min_side=min_side, max_side=max_side, stride=stride)

    if dataset == 'flickr32':
        Flickr32Dataset(root=dataset_root, transform=transform).describe_boxes()
