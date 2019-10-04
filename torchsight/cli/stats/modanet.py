"""Show stats about the Modanet dataset."""
import click


@click.command()
@click.option('-dr', '--dataset-root', required=True, type=click.Path(exists=True),
              help='The root directory of the dataset')
def modanet(dataset_root):
    """Show the stats of the modanet dataset."""
    from torchsight.datasets.modanet import ModanetDataset

    dataset = ModanetDataset(root=dataset_root)
    dataset.boxes_stats()
