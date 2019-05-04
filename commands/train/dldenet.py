"""CLI to train the DLDENet."""
import click

from torchsight.trainers import DLDENetTrainer


@click.command()
@click.argument('dataset-root', type=click.Path(exists=True))
# Currently we only have the COCO dataset for training
@click.option('--dataset', default='COCO', show_default=True, type=click.Choice(['COCO']))
@click.option('-b', '--batch-size', default=8, show_default=True)
@click.option('--resnet', default=50, show_default=True, help='The resnet backbone that the model must use.')
@click.option('--logs-dir', default='./logs', type=click.Path(exists=True), show_default=True,
              help='Where to store the checkpoints and descriptions.')
@click.option('-c', '--checkpoint', type=click.Path(exists=True), help='A checkpoint to resume the training from it.')
@click.option('--classes', default='',
              help='Indicate which classes (identified by its string label) must be used for the training. '
              'If no class is provided the trainer will use all the classes. Example: --classes "bear sheep airplane"')
@click.option('--optimizer', default='adabound', type=click.Choice(['adabound', 'sgd']), show_default=True,
              help='Set the optimizer that the trainer must use to train the model.')
@click.option('--not-normalize', is_flag=True,
              help='Avoid normalization of the embeddings in the classification module.')
@click.option('--device', default=None, help='The device that the model must use.')
def dldenet(dataset_root, dataset, batch_size, resnet, logs_dir, checkpoint, classes, optimizer, not_normalize,
            device):
    """Train the DLDENet with weighted classification vectors using the indicated dataset that
    contains is data in DATASET_ROOT directory."""
    classes = classes.split()
    n_classes = len(classes) if classes else 80

    DLDENetTrainer(
        hyperparameters={
            'model': {'resnet': resnet, 'classes': n_classes, 'normalize': not not_normalize},
            'datasets': {'root': dataset_root, 'class_names': classes},
            'dataloaders': {'batch_size': batch_size},
            'logger': {'dir': logs_dir},
            'checkpoint': {'dir': logs_dir},
            'optimizer': {'use': optimizer}
        },
        checkpoint=checkpoint,
        device=device
    ).train()
