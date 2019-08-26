"""CLI to train the DLDENet."""
import click


@click.command()
@click.option('--config', type=click.Path(exists=True), help='A JSON config file to load the configurations.'
              'If you provide this options all the other options are not used (only --device can be used).')
@click.option('--device', default=None, help='The device that the model must use.')
@click.option('--epochs', default=100, show_default=True)
def dldenet(config, device, epochs):
    """Train the DLDENet with a config in the indicated device for a number of epochs."""
    import json
    from torchsight.trainers import DLDENetTrainer

    with open(config, 'r') as file:
        hyperparameters = json.loads(file.read())

    DLDENetTrainer(hyperparameters=hyperparameters, device=device).train(epochs)


@click.command()
@click.option('-c', '--checkpoint', type=click.Path(exists=True), required=True)
@click.option('-dr', '--dataset-root', type=click.Path(exists=True), required=True)
@click.option('-b', '--batch-size', default=8, show_default=True, type=click.INT)
@click.option('--train-dir', default='.', show_default=True, type=click.Path(exists=True),
              help='Where to store the checkpoints.')
@click.option('--device', help='The device that the model must use.')
@click.option('--epochs', default=100, show_default=True)
def dldenet_from_checkpoint(dataset_root, checkpoint, batch_size, train_dir, device, epochs):
    """Get an instance of the trainer from the checkpoint and resume the exact same training
    with the dataset.

    You can only change things that will not affect the coherence of the training.
    """
    from torchsight.trainers import DLDENetTrainer

    new_params = {
        'datasets': {
            'coco': {'root': dataset_root},
            'logo32plus': {'root': dataset_root},
            'flickr32': {'root': dataset_root}
        }
    }

    if batch_size is not None:
        new_params['dataloaders'] = {'batch_size': batch_size}
    if train_dir is not None:
        new_params['checkpoint'] = {'dir': train_dir}

    DLDENetTrainer.from_checkpoint(checkpoint, new_params, device).train(epochs)
