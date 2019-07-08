"""Visualize the dataset Flickr32."""
import click


@click.command()
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True),
              help='The root directory of the dataset.')
@click.option('--dataset', default='trainval', type=click.Choice(['training', 'validation', 'trainval', 'test']))
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
@click.option('--classes', help='Visualize only this classes. Ex: "google esso"')
@click.option('--only-boxes', is_flag=True, help='Visualize only images with bounding boxes.')
def flickr32(dataset_root, dataset, no_shuffle, classes, only_boxes):
    """Visualize the images and annotations of the Flickr32 dataset that has its root directory
    at DATASET-ROOT."""
    import random
    from torchsight.datasets import Flickr32Dataset
    from torchsight.transforms.augmentation import AugmentDetection

    dataset = Flickr32Dataset(dataset_root, dataset, transform=AugmentDetection(evaluation=True, normalize=False),
                              classes=classes, only_boxes=only_boxes)

    length = len(dataset)
    print('Dataset length: {}'.format(length))
    print('Classes: {}'.format(list(dataset.class_to_label.keys())))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
