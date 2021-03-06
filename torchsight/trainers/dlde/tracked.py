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
            'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5},
            'means_update': 'batch',  # Could be 'batch' or 'manual'. See DirectionalClassifier module for more info.
            'means_lr': 0.1,  # Learning rate for the means in 'batch' mode
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
                'root': './datasets/logo32plus',
                'classes': None  # All classes
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
            'patience': 4,
            'threshold': 0.01
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
            means_update=hyperparameters['means_update'],
            means_lr=hyperparameters['means_lr'],
            pretrained=hyperparameters['pretrained']
        )

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
        classification_loss = classification_loss * weights['classification']
        regression_loss = regression_loss * weights['regression']

        loss = classification_loss + regression_loss

        # Log the classification and regression loss too:
        self.current_log['Class.'] = '{:.3f}'.format(float(classification_loss))
        self.current_log['Regr.'] = '{:.3f}'.format(float(regression_loss))
        # Also the mean of the weights and bias in the classification module:
        self.current_log['Cl. w'] = '{:.3f}'.format(float(self.model.classification.weight.mean()))
        self.current_log['Cl. b'] = '{:.3f}'.format(float(self.model.classification.bias.mean()))

        return loss

    def train(self, epochs=100, validate=True):
        """Train the model for the given epochs.

        If there is no checkpoint (i.e. checkpoint_epoch == 0) before training we make a loop over the training
        dataset to initialize the means with the random generated embeddings.

        Arguments:
            epochs (int): The number of epochs to train.
            validate (bool): If true it validates the model after each epoch using the validate method.
        """
        if self.hyperparameters['model']['means_update'] == 'manual':
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
                        self.logger.log({
                            'Initializing': None,
                            'Batch': '{}/{}'.format(batch_index + 1, n_batches),
                            'Time': '{:.3f} s'.format(time.time() - batch_start),
                            'Total': '{:.3f} s'.format(time.time() - start)
                        })
                    self.model.classification.update_means()
                    self.logger.log({
                        'Initializing': None,
                        'Means updated': None
                    })
                # Save the means as checkpoint in the epoch 0
                self.save(0)

        super().train(epochs, validate)

    def epoch_callback(self, epoch):
        """Update the means after each epoch.

        Arguments:
            epoch (int): The number of the epoch.
        """
        if self.hyperparameters['model']['means_update'] == 'manual':
            self.logger.log({
                'Training': None,
                'Epoch': '{}'.format(epoch),
                'Updating the model\'s means.': None
            })
            self.model.classification.update_means()
