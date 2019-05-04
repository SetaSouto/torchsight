"""Script to run a trainer.

You can modify the ResNet backbone and the batch size. For other hyperparameters modifications
you can edit the hyperparameters argument of the trainer. To see all the hyperparameters please
visit the original class definition.
"""
import argparse

from torchsight.trainers import (DLDENetTrainer,
                                 DLDENetWithTrackedMeansTrainer,
                                 RetinaNetTrainer)


def train():
    """Parse the arguments from the command line and start the training of the model."""
    parser = argparse.ArgumentParser(description='Train a model using COCO dataset.')
    parser.add_argument('model', help='The model to train. Supported "RetinaNet" and "DLDENet".')
    parser.add_argument('root', help='Root directory of the COCO dataset (cocoapi).')
    parser.add_argument('-b', '--batch-size', help='Batch size', default=8)
    parser.add_argument('--resnet', help='ResNet backbone to use.', default=18)
    parser.add_argument('--logs-dir', help='Directory where to save the logs and checkpoints.',
                        default='./logs')
    parser.add_argument('-c', '--checkpoint', help='Absolute path to the checkpoint to continue an old training')
    parser.add_argument('--device', help='The device to use in the training.')
    parser.add_argument('--optimizer', default='adabound',
                        help='Indicate the optimizer for the DLDENet. Could be adabound or sgd. Default: adabound')
    parser.add_argument('--not-normalize', action='store_const', const=True, default=False,
                        help='In the weighted DLDENet, avoid normalization of the embeddings.')

    args = parser.parse_args()

    classes = ()
    n_classes = len(classes) if classes else 80  # With an empty sequence it loads all the classes

    common_hyperparameters = {'datasets': {'root': args.root, 'class_names': classes},
                              'dataloaders': {'batch_size': int(args.batch_size)},
                              'model': {'resnet': int(args.resnet), 'classes': n_classes, 'normalize': not args.not_normalize},
                              'logger': {'dir': args.logs_dir},
                              'checkpoint': {'dir': args.logs_dir}}

    if args.model.lower() == 'retinanet':
        RetinaNetTrainer(
            hyperparameters=common_hyperparameters,
            checkpoint=args.checkpoint,
            device=args.device
        ).train()
    elif args.model.lower() == 'dldenetwithtrackedmeans':
        DLDENetWithTrackedMeansTrainer(
            hyperparameters={'optimizer': {'use': args.optimizer},
                             **common_hyperparameters},
            checkpoint=args.checkpoint,
            device=args.device
        ).train()
    elif args.model.lower() == 'dldenet':
        DLDENetTrainer(
            hyperparameters={'optimizer': {'use': args.optimizer},
                             **common_hyperparameters},
            checkpoint=args.checkpoint,
            device=args.device
        ).train()
    else:
        raise ValueError('The model "{}" is not supported.'.format(args.model))


if __name__ == '__main__':
    train()
