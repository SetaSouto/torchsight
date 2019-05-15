"""CLI to train the RetinaNet model."""
import click

from torchsight.trainers import RetinaNetTrainer


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
@click.option('--device', default=None, help='The device that the model must use.')
def retinanet(dataset_root, dataset, batch_size, resnet, logs_dir, checkpoint, classes, device):
    """Train a RetinaNet instance with the indicated dataset that contains its data in the
    DATASET_ROOT directory."""
    classes = classes.split()
    n_classes = len(classes) if classes else 80

    RetinaNetTrainer(
        hyperparameters={'datasets': {'root': dataset_root, 'class_names': classes},
                         'dataloaders': {'batch_size': batch_size},
                         'model': {'resnet': resnet, 'classes': n_classes},
                         'logger': {'dir': logs_dir},
                         'checkpoint': {'dir': logs_dir}},
        checkpoint=checkpoint,
        device=device
    ).train()
