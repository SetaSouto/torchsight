"""Visualize the dataset Flickr32."""
import click


@click.command()
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True),
              help='The root directory of the dataset.')
@click.option('--dataset', default='training', show_default=True,
              type=click.Choice(['training', 'validation', 'trainval']))
@click.option('--name', default='S2', type=click.Choice(['S1', 'S2', 'M']), show_default=True,
              help='The name of the instre dataset to visualize.')
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
def instre(dataset_root, dataset, name, no_shuffle):
    """Visualize the images and annotations of the instre dataset that has its root directory
    at DATASET-ROOT."""
    import random
    from torchvision.transforms import Compose
    from torchsight.datasets import InstreDataset
    from torchsight.transforms.detection import Resize

    dataset = InstreDataset(dataset_root, dataset=dataset, name=name, transform=Compose([
        Resize(min_side=384, max_side=512)
    ]))

    length = len(dataset)
    print('Dataset length: {}'.format(length))
    print('Classes: {}'.format(list(dataset.class_to_label.keys())))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
