"""RetinaNet Trainer."""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchsight import utils
from torchsight.optimizers import AdaBound, Lookahead, RAdam

from ..datasets import CocoDataset, Flickr32Dataset, Logo32plusDataset
from ..losses import FocalLoss
from ..models import RetinaNet
from ..transforms.augmentation import AugmentDetection
from .trainer import Trainer


class RetinaNetTrainer(Trainer):
    """RetinaNet trainer class."""

    ####################################
    ###           GETTERS            ###
    ####################################

    @staticmethod
    def get_base_hp():
        """Get the base hyperparameters for the trainer.

        Returns:
            JsonObject: with the base hyperparameters.
        """
        return Trainer.get_base_hp().merge({
            'model': {
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
                'pretrained': True,
                'evaluation': {
                    'threshold': 0.5,
                    'iou_threshold': 0.5
                }
            },
            'criterion': {
                'alpha': 0.25,
                'gamma': 2.0,
                'iou_thresholds': {
                    'background': 0.4,
                    'object': 0.5
                },
                'increase_foreground_by': 1e3,
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
                    'classes': None,
                },
                'flickr32': {
                    'root': './datasets/flickr32',
                    'brands': None,
                    'only_boxes': True,
                    'training': 'trainval',  # The name of the dataset to use for training
                    'validation': 'test'  # The name of the dataset to use for validation
                }
            },
            'dataloaders': {
                'batch_size': 1,
                'shuffle': True,
                'num_workers': 1
            },
            'optimizer': {
                'use': 'sgd',  # Which optimizer the trainer must use
                'adabound': {
                    'lr': 1e-3,  # Learning rate
                    'final_lr': 1  # When the optimizer change from Adam to SGD
                },
                # Lookahead is not an optimizer by itself, you must choose another of
                # the optimizers and set the 'use' flag to True in the lookahead params
                'lookahead': {
                    'use': False,
                    'k': 5,
                    'alpha': 0.5
                },
                'radam': {
                    'lr': 1e-3,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 0
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
            'transform': {
                'GaussNoise': {
                    'var_limit': (10, 50),
                    'p': 0.5
                },
                'GaussianBlur': {
                    'blur_limit': 0.7,
                    'p': 0.5
                },
                'RandomBrightnessContrast': {
                    'brightness_limit': 0.2,
                    'contrast_limit': 0.2,
                    'p': 0.5
                },
                'Rotate': {
                    'limit': 45,
                    'p': 0.5
                },
                'LongestMaxSize': {
                    'max_size': 512
                },
                'PadIfNeeded': {
                    'min_height': 512,
                    'min_width': 512
                },
                'RandomSizedBBoxSafeCrop': {
                    'height': 512,
                    'width': 512
                }
            }
        })

    def get_transform(self):
        """Initialize and get the transforms for the images.

        Arguments:
            params (dict): The dict with the params for the transforms.
                It must have 'resize' and 'normalize' args values.

        Returns:
            torchvision.transform.Compose: The Compose of the transformations.
        """
        return AugmentDetection(params=self.hyperparameters['transform'])

    def get_datasets(self):
        """Initialize and get the Coco datasets for training and evaluation.

        Returns:
            tuple: A Tuple with the torch.utils.data.Datasets for training and validation.
        """
        transform = self.get_transform()

        params = self.hyperparameters.datasets
        dataset = params.use

        if dataset == 'coco':
            params = params.coco

            n_classes = len(params.class_names)
            n_classes = 80 if n_classes == 0 else n_classes
            self.hyperparameters.model.classes = n_classes

            return (CocoDataset(root=params.root,
                                dataset=params.train,
                                classes_names=params.class_names,
                                transform=transform),
                    CocoDataset(root=params.root,
                                dataset=params.validation,
                                classes_names=params.class_names,
                                transform=transform))

        if dataset == 'logo32plus':
            params = params.logo32plus
            num_classes = len(params.classes) if params.classes is not None else 32

            self.hyperparameters.model.classes = num_classes

            return (Logo32plusDataset(**params, dataset='training', transform=transform),
                    Logo32plusDataset(**params, dataset='validation', transform=transform))

        if dataset == 'flickr32':
            params = params.flickr32
            num_classes = len(params.brands) if params.brands is not None else 32

            self.hyperparameters.model.classes = num_classes

            kwargs = {
                'root': params.root,
                'brands': params.brands,
                'only_boxes': params.only_boxes,
                'transform': transform
            }

            return (Flickr32Dataset(**kwargs, dataset=params.training),
                    Flickr32Dataset(**kwargs, dataset=params.validation))

    def get_dataloaders(self):
        """Initialize and get the dataloaders for the datasets.

        Returns:
            torch.utils.data.Dataloaders: The dataloader for the training dataset.
            torch.utils.data.Dataloaders: The dataloader for the validation dataset.
        """
        return utils.get_dataloaders(self.hyperparameters.dataloaders, self.dataset, self.valid_dataset)

    def get_model(self):
        """Initialize and get the RetinaNet.

        Returns:
            RetinaNet: The RetinaNet model.
        """
        return RetinaNet(**self.hyperparameters.model)

    def get_criterion(self):
        """Initialize and get the FocalLoss.

        Returns:
            torch.nn.Module: The FocalLoss.
        """
        return FocalLoss(**self.hyperparameters.criterion, device=self.device)

    def get_optimizer(self):
        """Returns the optimizer for the training.

        You can provide the optimizer that you want to use in the 'optimizer' hyperparameter
        changing the 'use' parameter and providing the name of the one that
        you want to use.

        Returns:
            Optimizer: The optimizer for the training.
        """
        params = self.hyperparameters.optimizer
        name = params.use.lower()

        if name == 'adabound':
            optimizer = AdaBound(self.model.parameters(), **params.adabound)
        elif name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), **params.sgd)
        elif name == 'radam':
            optimizer = RAdam(self.model.parameters(), **params.radam)
        else:
            raise ValueError('Cannot find the parameters for the optimizer "{}"'.format(params.use))

        # Check if we want to use the lookahead algorithm with the optimizer
        if params.lookahead.use:
            kwargs = {**params.lookahead}
            kwargs.pop('use')
            optimizer = Lookahead(optimizer, **kwargs)

        # Return the final optimizer for the trainer
        return optimizer

    def get_scheduler(self):
        """Get the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.
        """
        hyperparameters = self.hyperparameters.scheduler
        return ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=hyperparameters.factor,
            patience=hyperparameters.patience,
            verbose=True,
            threshold=hyperparameters.threshold
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

        classification_loss, regression_loss = self.criterion(anchors, regressions, classifications, annotations)
        del anchors, regressions, classifications, annotations

        weights = self.hyperparameters.criterion.weights
        classification_loss *= weights.classification
        regression_loss *= weights.regression

        loss = classification_loss + regression_loss

        # Log the classification and regression loss too:
        self.current_log['Class.'] = '{:.5f}'.format(float(classification_loss))
        self.current_log['Regr.'] = '{:.5f}'.format(float(regression_loss))
        self.current_log['Pos'] = '{:.4f}'.format(float(self.criterion.pos_loss * weights.classification))
        self.current_log['Neg'] = '{:.4f}'.format(float(self.criterion.neg_loss * weights.classification))

        return loss

    def backward(self, loss):
        """Do the backward pass over the network.

        Arguments:
            loss (torch.Tensor): The loss value computed during the forward pass.
        """
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

    def eval(self):
        """Put the model in evaluation mode."""
        params = self.hyperparameters.model.evaluation
        self.model.eval(threshold=params.threshold, iou_threshold=params.iou_threshold, loss=True)
