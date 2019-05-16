"""CLI to train the DLDENet."""
import click

from torchsight.trainers import DLDENetTrainer


@click.command()
@click.argument('dataset-root', type=click.Path(exists=True))
# Currently we only have the COCO dataset for training
@click.option('--dataset', default='coco', show_default=True, type=click.Choice(['coco', 'logo32plus']))
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
    contains is data in DATASET_ROOT directory.

    You can use a checkpoint to resume the training but is not a good practice, because you can change the hyperparams
    of the model get in some troubles like changing the resnet backbone.
    To avoid this you can use the subcommand 'dldenet-from-checkpoint' instead.
    """
    classes = classes.split()

    DLDENetTrainer(
        hyperparameters={
            'model': {'resnet': resnet, 'normalize': not not_normalize},
            'datasets': {
                'use': dataset,
                'coco': {'root': dataset_root, 'class_names': classes},
                'logo32plus': {'root': dataset_root}
            },
            'dataloaders': {'batch_size': batch_size},
            'logger': {'dir': logs_dir},
            'checkpoint': {'dir': logs_dir},
            'optimizer': {'use': optimizer}
        },
        checkpoint=checkpoint,
        device=device
    ).train()


@click.command()
@click.argument('dataset-root', type=click.Path(exists=True))
@click.argument('checkpoint', type=click.Path(exists=True))
@click.option('-b', '--batch-size', type=click.INT)
@click.option('--logs-dir', type=click.Path(exists=True), help='Where to store the checkpoints and descriptions.')
@click.option('--device', help='The device that the model must use.')
def dldenet_from_checkpoint(dataset_root, checkpoint, batch_size, logs_dir, device):
    """Get an instance of the trainer from the checkpoint CHECKPOINT and resume the exact same training
    with the dataset that contains its data in DATASET_ROOT.

    You can only change things that will not affect the coherence of the training.
    """
    new_params = {'datasets': {'root': dataset_root}}

    if batch_size is not None:
        new_params['dataloaders'] = {'batch_size': batch_size}
    if logs_dir is not None:
        new_params['logger'] = {'dir': logs_dir}
        new_params['checkpoint'] = {'dir': logs_dir}

    DLDENetTrainer.from_checkpoint(checkpoint, new_params, device).train()
