"""DLDENet trainer for the tracked means version."""
import time

import torch

from torchsight.models import DLDENetWithTrackedMeans
from torchsight.optimizers import AdaBound
from ..retinanet import RetinaNetTrainer


class DLDENetWithTrackedMeansTrainer(RetinaNetTrainer):
    """Deep Local Directional Embedding with tracked means trainer.

    As the architecture is very similar with RetinaNet we use the same trainer and only
    override some attributes and methods. For more information please read the RetinaNet
    trainer documentation.
    """
    # Base hyperparameters, can be replaced in the initialization of the trainer
    hyperparameters = {
        'model': {
            'resnet': 18,
            'features': {
                'pyramid': 256,
                'regression': 256,
                'classification': 256
            },
            'anchors': {
                'sizes': [32, 64, 128, 256, 512],
                'scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)],
                'ratios': [0.5, 1, 2]
            },
            'embedding_size': 256,
            'pretrained': True,
            'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5}
        },
        'criterion': {
            'alpha': 0.25,
            'gamma': 2.0,
            'iou_thresholds': {'background': 0.4, 'object': 0.5},
            # Weight of each loss. See train method.
            'weights': {'classification': 1e3, 'regression': 1}
        },
        'datasets': {
            'use': 'coco',
            'coco': {
                'root': './datasets/coco',
                'class_names': (),  # () indicates all classes
                'train': 'train2017',
                'validation': 'val2017'
            },
            'logo32plus': {
                'root': './datasets/logo32plus'
            }
        },
        'dataloaders': {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 8
        },
        'optimizer': {
            'use': 'adabound',  # Which optimizer the trainer must use
            'adabound': {
                'lr': 1e-3,  # Learning rate
                'final_lr': 0.1  # When the optimizer must change from Adam to SGD
            },
            'sgd': {
                'lr': 1e-2,
                'momentum': 0.9,
                'weight_decay': 1e-4
            }
        },
        'scheduler': {
            'factor': 0.1,
            'patience': 2,
            'threshold': 0.1
        },
        'transforms': {
            'resize': {
                'min_side': 384,
                'max_side': 512,
                'stride': 128
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }

    def get_model(self):
        """Initialize and get a DLDENet model instance."""
        hyperparameters = self.hyperparameters['model']
        hyperparameters['assignation_thresholds'] = self.hyperparameters['criterion']['iou_thresholds']

        return DLDENetWithTrackedMeans(
            classes=hyperparameters['classes'],
            resnet=hyperparameters['resnet'],
            features=hyperparameters['features'],
            anchors=hyperparameters['anchors'],
            embedding_size=hyperparameters['embedding_size'],
            assignation_thresholds=hyperparameters['assignation_thresholds'],
            pretrained=hyperparameters['pretrained']
        )

    def get_optimizer(self):
        """Returns the optimizer for the training.

        The extended RetinaNetTrainer only has SGD as an optimizer, this trainer also
        includes the [AdaBound optimizer](https://github.com/Luolc/AdaBound) that it's
        supposed to be as fast as Adam and as good as SGD.

        You can provide the optimizer that you want to use in the 'optimizer' hyperparameter
        changing the 'use' parameter and providing the name of the one that
        you want to use.

        Returns:
            AdaBound: The adabound optimizer for the training.
        """
        params = self.hyperparameters['optimizer']
        optimizer = params['use'].lower()

        if optimizer == 'adabound':
            params = params['adabound']
            return AdaBound(self.model.parameters(), lr=params['lr'], final_lr=params['final_lr'])

        if optimizer == 'sgd':
            params = params['sgd']
            return torch.optim.SGD(self.model.parameters(),
                                   lr=params['lr'],
                                   momentum=params['momentum'],
                                   weight_decay=params['weight_decay'])

        raise ValueError('Cannot find the parameters for the optimizer "{}"'.format(params['use']))

    def forward(self, *args):
        """Forward pass through the network and loss computation.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        images, annotations, *_ = args
        images, annotations = images.to(self.device), annotations.to(self.device)

        anchors, regressions, classifications = self.model(images, annotations)
        del images

        classification_loss, regression_loss = self.criterion(anchors, regressions, classifications, annotations)
        del anchors, regressions, classifications, annotations

        weights = self.hyperparameters['criterion']['weights']
        classification_loss *= weights['classification']
        regression_loss *= weights['regression']

        loss = classification_loss + regression_loss

        # Log the classification and regression loss too:
        self.current_log['Class.'] = float(classification_loss)
        self.current_log['Regr.'] = float(regression_loss)

        return loss

    def train(self, epochs=100, validate=True):
        """Train the model for the given epochs.

        If there is no checkpoint (i.e. checkpoint_epoch == 0) before training we make a loop over the training
        dataset to initialize the means with the random generated embeddings.

        Arguments:
            epochs (int): The number of epochs to train.
            validate (bool): If true it validates the model after each epoch using the validate method.
        """
        self.model.to(self.device)
        self.model.train()
        n_batches = len(self.dataloader)

        if self.checkpoint is None:
            # Initialize the means of the classes
            with torch.no_grad():
                start = time.time()
                for batch_index, (images, annotations, *_) in enumerate(self.dataloader):
                    batch_start = time.time()
                    self.model(images.to(self.device), annotations.to(self.device), initializing=True)
                    print('[Initializing] [Batch {}/{}] [Batch {:.3f} s] [Total {:.3f} s]'.format(
                        batch_index + 1, n_batches, time.time() - batch_start, time.time() - start))
                self.model.update_means()
                print('[Initializing] Means updated.')
            # Save the means as checkpoint in the epoch 0
            self.save(0)

        super().train(epochs, validate)

    def epoch_callback(self, epoch):
        """Update the means after each epoch.

        Arguments:
            epoch (int): The number of the epoch.
        """
        print("[Training] [Epoch {}] Updating the model's means.".format(epoch))
        self.model.classification.update_means()
