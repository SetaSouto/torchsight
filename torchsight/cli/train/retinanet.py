"""CLI to train the RetinaNet model."""
import click


@click.command()
@click.option('--config', type=click.Path(exists=True), help='An optional configuration file with the hyperparameters. '
              'It will block all the other options, only --device and --checkpoint will work.')
@click.option('-d', '--dataset', default='coco', show_default=True, type=click.Choice(['coco', 'flickr32', 'logo32plus']))
@click.option('-dr', '--dataset-root', type=click.Path(exists=True))
@click.option('-b', '--batch-size', default=8, show_default=True)
@click.option('--resnet', default=50, show_default=True, help='The resnet backbone that the model must use.')
@click.option('--logs-dir', default='.', type=click.Path(), show_default=True,
              help='Where to store the checkpoints and descriptions.')
@click.option('-c', '--checkpoint', type=click.Path(exists=True), help='A checkpoint to resume the training from it.')
@click.option('--classes', default=None,
              help='Indicate which classes (identified by its string label) must be used for the training. '
              'If no class is provided the trainer will use all the classes. Example: --classes "bear sheep airplane"')
@click.option('--device', default=None, help='The device that the model must use.')
def retinanet(config, dataset, dataset_root, batch_size, resnet, logs_dir, checkpoint, classes, device):
    """Train a RetinaNet instance with the indicated dataset that contains its data in the
    DATASET_ROOT directory."""
    from torchsight.trainers import RetinaNetTrainer

    if config is None:
        classes = classes.split() if classes is not None else None
        hyperparameters = {
            'datasets': {
                'use': dataset,
                'coco': {
                    'root': dataset_root,
                    'class_names': classes,
                },
                'logo32plus': {
                    'root': dataset_root,
                    'classes': classes,
                },
                'flickr32': {
                    'root': dataset_root,
                    'classes': classes,
                }
            },
            'dataloaders': {
                'batch_size': batch_size
            },
            'model': {
                'resnet': resnet,
            },
            'logger': {
                'dir': logs_dir
            },
            'checkpoint': {
                'dir': logs_dir
            }
        }
    else:
        import json
        with open(config, 'r') as file:
            hyperparameters = json.loads(file.read())

    RetinaNetTrainer(
        hyperparameters=hyperparameters,
        checkpoint=checkpoint,
        device=device
    ).train()
