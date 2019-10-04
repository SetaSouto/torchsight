"""Visualize the dataset Modanet."""
import click


@click.command()
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True),
              help='The root directory of the dataset.')
@click.option('--validation', is_flag=True, help='Load the validation dataset.')
@click.option('--no-shuffle', is_flag=True, help='Show the images in order and not randomly.')
def modanet(dataset_root, validation, no_shuffle):
    """Visualize the images and annotations of the Modanet dataset."""
    import random
    from torchsight.datasets.modanet import ModanetDataset
    from torchsight.transforms.augmentation import AugmentDetection

    dataset = ModanetDataset(
        root=dataset_root,
        transform=AugmentDetection(evaluation=True, normalize=False),
        valid=validation
    )

    length = len(dataset)

    print(f'Dataset length: {length}')
    print('Classes:')
    for label, name in dataset.label_to_class.items():
        print(f'{str(label).ljust(2)} {name}')

    indexes = list(range(length))

    if not no_shuffle:
        random.shuffle(indexes)

    for i in indexes:
        dataset.visualize(i)
