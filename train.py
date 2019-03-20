"""Script to run a trainer.

You can modify the ResNet backbone and the batch size. For other hyperparameters modifications
you can edit the hyperparameters argument of the trainer. To see all the hyperparameters please
visit the original class definition.
"""
import argparse

from torchsight.trainers import RetinaNetTrainer

PARSER = argparse.ArgumentParser(description='Train a RetinaNet model using COCO dataset.')
PARSER.add_argument('model', help='The model to train. Currently only supported "RetinaNet".', default='RetinaNet')
PARSER.add_argument('-r', '--root', help='Root directory of the COCO dataset (cocoapi).', required=True)
PARSER.add_argument('-b', '--batch-size', help='Batch size', default=1)
PARSER.add_argument('--resnet', help='ResNet backbone to use.', default=50)
PARSER.add_argument('-l', '--logs-dir', help='Directory where to save the logs and checkpoints.',
                    default='./logs')
PARSER.add_argument('-c', '--checkpoint', help='Absolute path to the checkpoint to continue an old training')

ARGUMENTS = PARSER.parse_args()

if ARGUMENTS.model.lower() == 'retinanet':
    RetinaNetTrainer(
        hyperparameters={
            'RetinaNet': {'resnet': int(ARGUMENTS.resnet)},
            'datasets': {'root': ARGUMENTS.root},
            'dataloaders': {'batch_size': int(ARGUMENTS.batch_size)}
        },
        logs_dir=ARGUMENTS.logs_dir,
        checkpoint=ARGUMENTS.checkpoint
    ).train()
else:
    raise ValueError('The model "{}" is not supported.'.format(ARGUMENTS.model))

