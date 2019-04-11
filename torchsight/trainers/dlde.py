"""DLDENet trainer"""
import torch

from ..models import DLDENet
from .retinanet import RetinaNetTrainer


class DLDENetTrainer(RetinaNetTrainer):
    """Deep Local Directional Embedding trainer.

    As the architecture is very similar with RetinaNet we use the same trainer and only
    override some attributes and methods. For more information please read the RetinaNet
    trainer documentation.
    """

    # Base hyperparameters, can be replaced in the initialization of the trainer:
    # >>> RetinaNetTrainer(hyperparameters={'RetinaNet': {'classes': 1}})
    hyperparameters = {
        'DLDENet': {
            'classes': 80,
            'resnet': 50,
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
            'concentration': 15,
            # Keep in mind that this thresholds must be the same as in the FocalLoss
            'assignation_thresholds': {'object': 0.5, 'background': 0.4},
            'pretrained': True,
            'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5}
        },
        'FocalLoss': {
            'alpha': 0.25,
            'gamma': 2.0,
            'iou_thresholds': {'background': 0.4, 'object': 0.5},
            # Weight of each loss. See train method.
            'weights': {'classification': 1e5, 'regression': 1}
        },
        'datasets': {
            'root': './datasets/coco',
            'class_names': (),  # Empty tuple indicates all classes
            'train': 'train2017',
            'validation': 'val2017'
        },
        'dataloaders': {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 8
        },
        'optimizer': {
            'learning_rate': 1e-2,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'factor': 0.1,
            'patience': 2,
            'threshold': 0.1
        },
        'transforms': {
            'resize': {
                'min_side': 640,
                'max_side': 1024,
                'stride': 128
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }

    def get_model_hyperparameters(self):
        return self.hyperparameters['DLDENet']

    def get_model(self):
        """Initialize and get a DLDENet model instance."""
        hyperparameters = self.get_model_hyperparameters()
        return DLDENet(
            classes=hyperparameters['classes'],
            resnet=hyperparameters['resnet'],
            features=hyperparameters['features'],
            anchors=hyperparameters['anchors'],
            embedding_size=hyperparameters['embedding_size'],
            concentration=hyperparameters['concentration'],
            assignation_thresholds=hyperparameters['assignation_thresholds'],
            pretrained=hyperparameters['pretrained']
        )

    def forward(self, images, annotations):
        """Forward pass through the network during training."""
        return self.model(images, annotations)

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

        if self.checkpoint_epoch == 0:
            # Initialize the means of the classes
            with torch.no_grad():
                for batch_index, (images, annotations, *_) in enumerate(self.dataloader):
                    print('[Initializing] [Batch {}/{}]'.format(batch_index + 1, n_batches))
                    self.model(images.to(self.device), annotations.to(self.device), initializing=True)
                self.model.update_means()
                print('[Initializing] Means updated.')

        def callback(epoch):
            print("[Training] [Epoch {}] Updating the model's means.".format(epoch))
            self.model.update_means()

        super(DLDENetTrainer, self).train(epochs, validate, callback)
