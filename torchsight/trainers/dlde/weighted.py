"""DLDENet trainer for the weighted version."""
import torch

from torchsight.losses import DLDENetLoss
from torchsight.models import DLDENet
from torchsight.optimizers import AdaBound

from ..retinanet import RetinaNetTrainer


class DLDENetTrainer(RetinaNetTrainer):
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
            'normalize': True,
            'pretrained': True,
            'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5},
            'weighted_bias': False,
            'fixed_bias': 0,
            'increase_norm_by': None
        },
        'criterion': {
            'alpha': 0.25,
            'gamma': 2.0,
            'sigma': 3.0,
            'iou_thresholds': {'background': 0.4, 'object': 0.5},
            # Weight of each loss. See train method.
            'weights': {'classification': 1e4, 'regression': 1, 'similarity': 1},
            'soft': False,  # Apply soft loss weighted by the IoU
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
                'classes': None,
            },
            'flickr32': {
                'root': './datasets/flickr32',
                'classes': None,
                'only_boxes': True,
                'training': 'trainval',  # The name of the dataset to use for training
                'validation': 'test'  # The name of the dataset to use for validation
            }
        },
        'dataloaders': {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 8
        },
        'optimizer': {
            'use': 'sgd',  # Which optimizer the trainer must use
            'adabound': {
                'lr': 1e-3,  # Learning rate
                'final_lr': 1  # When the optimizer change from Adam to SGD
            },
            'sgd': {
                'lr': 1e-2,
                'momentum': 0.9,
                'weight_decay': 1e-4
            }
        },
        'scheduler': {
            'factor': 0.1,
            'patience': 5,
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

    ####################################
    ###           GETTERS            ###
    ####################################

    def get_model(self):
        """Initialize and get a DLDENet model instance."""
        hyperparameters = self.hyperparameters['model']
        return DLDENet(
            classes=hyperparameters['classes'],
            resnet=hyperparameters['resnet'],
            features=hyperparameters['features'],
            anchors=hyperparameters['anchors'],
            embedding_size=hyperparameters['embedding_size'],
            normalize=hyperparameters['normalize'],
            pretrained=hyperparameters['pretrained'],
            device=self.device,
            weighted_bias=hyperparameters['weighted_bias'],
            fixed_bias=hyperparameters['fixed_bias'],
            increase_norm_by=hyperparameters['increase_norm_by'],
        )

    def get_criterion(self):
        """Get the criterion to use to train the model.

        Returns:
            DLDENetLoss: The unified loss between FocalLoss and Cosine similarity Loss.
        """
        params = self.hyperparameters['criterion']

        return DLDENetLoss(
            alpha=params['alpha'],
            gamma=params['gamma'],
            sigma=params['sigma'],
            iou_thresholds=params['iou_thresholds'],
            soft=params['soft'],
            device=self.device
        )

    ####################################
    ###           METHODS            ###
    ####################################

    def forward(self, *args):
        """Forward pass through the network and loss computation.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        images, annotations, *_ = args
        images, annotations = images.to(self.device), annotations.to(self.device)

        anchors, regressions, classifications = self.model(images)
        del images

        losses = self.criterion(anchors, regressions, classifications, annotations, self.model)
        del anchors, regressions, classifications, annotations

        classification, regression, similarity = losses

        # Amplify the losses according to the criterion weights
        weights = self.hyperparameters['criterion']['weights']
        classification *= weights['classification']
        regression *= weights['regression']
        similarity *= weights['similarity']

        loss = classification + regression + similarity

        # Log the classification and regression loss too:
        self.current_log['Class.'] = '{:.4f}'.format(float(classification))
        self.current_log['Regr.'] = '{:.4f}'.format(float(regression))
        self.current_log['Simil.'] = '{:.4f}'.format(float(similarity))
        # Log the mean norm of the weights in the classification module and their biases
        self.current_log['w-norm'] = '{:.4f}'.format(float(self.model.classification.weights.norm(dim=0).mean()))
        if self.model.classification.weighted_bias:
            self.current_log['bias'] = '{:.4f}'.format(float(self.model.classification.bias.mean()))

        return loss
