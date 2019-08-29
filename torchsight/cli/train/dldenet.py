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
@click.option('-ch', '--checkpoint', type=click.Path(exists=True), required=True)
@click.option('-co', '--config', type=click.Path(exists=True),
              help='The new config to use to override the hyperparameters of the checkpoint')
@click.option('--device', help='The device that the model must use.')
@click.option('--epochs', default=100, show_default=True)
def dldenet_from_checkpoint(checkpoint, config, device, epochs):
    """Get an instance of the trainer from the checkpoint and resume the exact same training
    with the dataset and optionally change any hyperparameter with the new config provided.
    """
    import json
    from torchsight.trainers import DLDENetTrainer

    new_params = {}
    if config is not None:
        with open(config, 'r') as file:
            new_params = json.loads(file.read())

    DLDENetTrainer.from_checkpoint(checkpoint, new_params, device).train(epochs)
