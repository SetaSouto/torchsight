"""Visualize the dataset Flickr32."""
import click


@click.command()
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True),
              help='The root directory of the dataset.')
@click.option('--dataset', default='training', type=click.Choice(['training', 'validation', 'trainval', 'test']))
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
@click.option('--classes', help='Visualize only this classes. Ex: "google esso"')
@click.option('--only-boxes', is_flag=True, help='Visualize only images with bounding boxes.')
def flickr32(dataset_root, dataset, no_shuffle, classes, only_boxes):
    """Visualize the images and annotations of the Flickr32 dataset that has its root directory
    at DATASET-ROOT."""
    import random
    from torchvision.transforms import Compose
    from torchsight.datasets import Flickr32Dataset
    from torchsight.transforms.detection import Resize

    dataset = Flickr32Dataset(dataset_root, dataset, transform=Compose([
        Resize(min_side=384, max_side=512)
    ]), classes=classes, only_boxes=only_boxes)

    length = len(dataset)
    print('Dataset length: {}'.format(length))
    print('Classes: {}'.format(list(dataset.class_to_label.keys())))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
