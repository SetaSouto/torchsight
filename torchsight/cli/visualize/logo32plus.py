"""Visualize the dataset Logo32plus."""
import click


@click.command()
@click.argument('dataset-root')
@click.option('--dataset', default='training', type=click.Choice(['training', 'validation', 'both']))
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
def logo32plus(dataset_root, dataset, no_shuffle):
    """Visualize the images and annotations of the Logo32plus dataset that has its root directory
    at DATASET-ROOT."""
    import random
    from torchvision.transforms import Compose
    from torchsight.datasets import Logo32plusDataset
    from torchsight.transforms.detection import Resize

    dataset = Logo32plusDataset(dataset_root, dataset, transform=Compose([
        Resize(min_side=384, max_side=512)
    ]))

    length = len(dataset)
    print('Dataset length: {}'.format(length))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
