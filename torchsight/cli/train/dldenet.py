"""CLI to train the DLDENet."""
import click

from torchsight.trainers import DLDENetTrainer, DLDENetWithTrackedMeansTrainer


@click.command()
@click.option('-d', '--dataset', default='coco', show_default=True, type=click.Choice(['coco', 'logo32plus', 'flickr32']))
@click.option('-dr', '--dataset-root', type=click.Path(exists=True), required=True)
@click.option('-b', '--batch-size', default=8, show_default=True)
@click.option('--resnet', default=50, show_default=True, help='The resnet backbone that the model must use.')
@click.option('--fixed-bias', default=-0.5, show_default=True, help='The fixed bias for the classification module.')
@click.option('--logs-dir', default='./logs', type=click.Path(exists=True), show_default=True,
              help='Where to store the checkpoints and descriptions.')
@click.option('--classes', default='',
              help='Indicate which classes (identified by its string label) must be used for the training. '
              'If no class is provided the trainer will use all the classes. Example: --classes "bear sheep airplane"')
@click.option('--optimizer', default='adabound', type=click.Choice(['adabound', 'sgd']), show_default=True,
              help='Set the optimizer that the trainer must use to train the model.')
@click.option('--adabound-lr', default=1e-3, show_default=True, help='The learning rate for the starting in Adabound.')
@click.option('--adabound-final-lr', default=1, show_default=True,
              help='The final learning rate when Adabound transform to SGD.')
@click.option('--scheduler-factor', default=0.1, show_default=True,
              help='The factor to scale the LR.')
@click.option('--scheduler-patience', default=5, show_default=True,
              help='Hoy many epochs without relative improvement the scheduler must wait.')
@click.option('--scheduler-threshold', default=0.01, show_default=True,
              help='The relative threshold that indicates an improvement for the scheduler.')
@click.option('--anchors-sizes', default='20 40 80 160 320', show_default=True)
@click.option('--not-normalize', is_flag=True,
              help='Avoid normalization of the embeddings in the classification module. Only available without tracked means.')
@click.option('--device', default=None, help='The device that the model must use.')
@click.option('--tracked-means', is_flag=True, help='Use the version that tracks the means.')
@click.option('--soft-criterion', is_flag=True, help='Use soft assignment in the Loss.')
@click.option('--means-update', default='batch', type=click.Choice(['batch', 'manual']), show_default=True,
              help='Update type for the means in the tracked version. See DirectionalClassification module for more info.')
@click.option('--means-lr', default=0.1, show_default=True, help='The learning rate for the "batch" means update method.')
@click.option('--num-workers', default=8, show_default=True)
@click.option('--epochs', default=100, show_default=True)
def dldenet(dataset_root, dataset, batch_size, resnet, fixed_bias, logs_dir, classes, optimizer,
            adabound_lr, adabound_final_lr, scheduler_factor, scheduler_patience, scheduler_threshold,
            anchors_sizes, num_workers,
            not_normalize, device, tracked_means, soft_criterion, epochs, means_update, means_lr):
    """Train the DLDENet with weighted classification vectors using the indicated dataset that
    contains is data in DATASET_ROOT directory.
    """
    classes = classes.split()
    hyperparameters = {
        'model': {
            'resnet': resnet,
            'normalize': not not_normalize,
            'means_update': means_update,
            'means_lr': means_lr,
            'fixed_bias': fixed_bias,
            'anchors': {
                'sizes': [int(size) for size in anchors_sizes.split()],
            },
        },
        'criterion': {
            'soft': soft_criterion
        },
        'datasets': {
            'use': dataset,
            'coco': {'root': dataset_root, 'class_names': classes},
            'logo32plus': {'root': dataset_root, 'classes': classes if classes else None},
            'flickr32': {'root': dataset_root, 'classes': classes if classes else None}
        },
        'dataloaders': {
            'batch_size': batch_size,
            'num_workers': num_workers,
        },
        'logger': {'dir': logs_dir},
        'checkpoint': {'dir': logs_dir},
        'scheduler': {
            'factor': scheduler_factor,
            'patience': scheduler_patience,
            'threshold': scheduler_threshold,
        },
        'optimizer': {
            'use': optimizer,
            'adabound': {
                'lr': adabound_lr,
                'final_lr': adabound_final_lr,
            },
        },
    }
    params = {'hyperparameters': hyperparameters, 'device': device}

    if tracked_means:
        DLDENetWithTrackedMeansTrainer(**params).train(epochs)
    else:
        DLDENetTrainer(**params).train(epochs)


@click.command()
@click.option('-c', '--checkpoint', type=click.Path(exists=True), required=True)
@click.option('-dr', '--dataset-root', type=click.Path(exists=True), required=True)
@click.option('-b', '--batch-size', default=8, show_default=True, type=click.INT)
@click.option('--logs-dir', default='./logs', show_default=True, type=click.Path(exists=True),
              help='Where to store the checkpoints and descriptions.')
@click.option('--device', help='The device that the model must use.')
@click.option('--epochs', default=100, show_default=True)
@click.option('--tracked-means', is_flag=True, help='Use the tracked means version.')
def dldenet_from_checkpoint(dataset_root, checkpoint, batch_size, logs_dir, device, epochs, tracked_means):
    """Get an instance of the trainer from the checkpoint CHECKPOINT and resume the exact same training
    with the dataset that contains its data in DATASET_ROOT.

    You can only change things that will not affect the coherence of the training.
    """
    new_params = {
        'datasets': {
            'coco': {'root': dataset_root},
            'logo32plus': {'root': dataset_root},
            'flickr32': {'root': dataset_root}
        }
    }

    if batch_size is not None:
        new_params['dataloaders'] = {'batch_size': batch_size}
    if logs_dir is not None:
        new_params['logger'] = {'dir': logs_dir}
        new_params['checkpoint'] = {'dir': logs_dir}

    if tracked_means:
        DLDENetWithTrackedMeansTrainer.from_checkpoint(checkpoint, new_params, device).train(epochs)
    else:
        DLDENetTrainer.from_checkpoint(checkpoint, new_params, device).train(epochs)
