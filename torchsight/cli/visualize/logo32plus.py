"""Visualize the dataset Logo32plus."""
import click


@click.command()
@click.argument('dataset-root')
@click.option('--dataset', default='both', type=click.Choice(['training', 'validation', 'both']))
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
@click.option('--classes', help='Visualize only this classes. Ex: "google esso"')
def logo32plus(dataset_root, dataset, no_shuffle, classes):
    """Visualize the images and annotations of the Logo32plus dataset that has its root directory
    at DATASET-ROOT."""
    import random
    from torchsight.datasets import Logo32plusDataset
    from torchsight.transforms.augmentation import AugmentDetection

    dataset = Logo32plusDataset(dataset_root, dataset, transform=AugmentDetection(
        evaluation=True, normalize=False), classes=classes)

    length = len(dataset)
    print('Dataset length: {}'.format(length))
    print('Classes: {}'.format(list(dataset.class_to_label.keys())))
    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
