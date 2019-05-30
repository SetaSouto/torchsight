"""RetinaNet Trainer."""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from ..datasets import CocoDataset, Logo32plusDataset
from ..losses import FocalLoss
from ..models import RetinaNet
from ..transforms.detection import Normalize, Resize, ToTensor
from .trainer import Trainer


class RetinaNetTrainer(Trainer):
    """RetinaNet trainer class."""
    # Base hyperparameters, can be replaced in the initialization of the trainer:
    # >>> RetinaNetTrainer(hyperparameters={'RetinaNet': {'classes': 1}})
    hyperparameters = {
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
            # Weight of each loss. See train method.
            'weights': {'classification': 1e5, 'regression': 1}
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
            'num_workers': 1
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

    ####################################
    ###           GETTERS            ###
    ####################################

    @staticmethod
    def get_transform(params):
        """Initialize and get the transforms for the images.

        Arguments:
            params (dict): The dict with the params for the transforms.
                It must have 'resize' and 'normalize' args values.

        Returns:
            torchvision.transform.Compose: The Compose of the transformations.
        """
        return transforms.Compose([
            Resize(**params['resize']),
            ToTensor(),
            Normalize(**params['normalize'])
        ])

    def get_datasets(self):
        """Initialize and get the Coco datasets for training and evaluation.

        Returns:
            tuple: A Tuple with the torch.utils.data.Datasets for training and validation.
        """
        transform = self.get_transform(self.hyperparameters['transforms'])

        params = self.hyperparameters['datasets']
        dataset = params['use']

        if dataset == 'coco':
            params = params['coco']

            n_classes = len(params['class_names'])
            n_classes = 80 if n_classes == 0 else n_classes
            self.hyperparameters['model']['classes'] = n_classes

            return (CocoDataset(root=params['root'],
                                dataset=params['train'],
                                classes_names=params['class_names'],
                                transform=transform),
                    CocoDataset(root=params['root'],
                                dataset=params['validation'],
                                classes_names=params['class_names'],
                                transform=transform))

        if dataset == 'logo32plus':
            params = params['logo32plus']
            num_classes = len(params['classes']) if params['classes'] is not None else 32

            self.hyperparameters['model']['classes'] = num_classes

            return (Logo32plusDataset(**params, dataset='training', transform=transform),
                    Logo32plusDataset(**params, dataset='validation', transform=transform))

    def get_dataloaders(self):
        """Initialize and get the dataloaders for the datasets.

        Returns:
            torch.utils.data.Dataloaders: The dataloader for the training dataset.
            torch.utils.data.Dataloaders: The dataloader for the validation dataset.
        """
        def collate(data):
            """Custom collate function to join the different images with its different annotations.

            Why is this important?
            Because as each image could contain different amounts of annotated objects the tensor
            for the batch could not be created, so we need to "fill" the annotations tensors with -1
            to give them the same shapes and stack them.
            Why -1?
            Because the FocalLoss could interpret that label and ingore it for the loss.

            Also it pads the images so all has the same size.

            Arguments:
                data (sequence): Sequence of tuples as (image, annotations, *_).

            Returns:
                torch.Tensor: The images.
                    Shape:
                        (batch size, channels, height, width)
                torch.Tensor: The annotations.
                    Shape:
                        (batch size, biggest amount of annotations, 5)
            """
            images = [image for image, *_ in data]
            max_width = max([image.shape[-1] for image in images])
            max_height = max([image.shape[-2] for image in images])

            def pad_image(image):
                aux = torch.zeros((image.shape[0], max_height, max_width))
                aux[:, :image.shape[1], :image.shape[2]] = image
                return aux

            images = torch.stack([pad_image(image) for image, *_ in data], dim=0)

            max_annotations = max([annotations.shape[0] for _, annotations, *_ in data])

            def fill_annotations(annotations):
                aux = torch.ones((max_annotations, 5))
                aux *= -1
                aux[:annotations.shape[0], :] = annotations
                return aux

            annotations = torch.stack([fill_annotations(a) for _, a, *_ in data], dim=0)
            return images, annotations

        hyperparameters = {**self.hyperparameters['dataloaders'], 'collate_fn': collate}

        return (
            DataLoader(dataset=self.dataset, **hyperparameters),
            DataLoader(dataset=self.valid_dataset, **hyperparameters)
        )

    def get_model(self):
        """Initialize and get the RetinaNet.

        Returns:
            torch.nn.Module: The RetinaNet model.
        """
        hyperparameters = self.hyperparameters['model']

        return RetinaNet(
            classes=hyperparameters['classes'],
            resnet=hyperparameters['resnet'],
            features=hyperparameters['features'],
            anchors=hyperparameters['anchors'],
            pretrained=hyperparameters['pretrained']
        )

    def get_criterion(self):
        """Initialize and get the FocalLoss.

        Returns:
            torch.nn.Module: The FocalLoss.
        """
        hyperparameters = self.hyperparameters['criterion']

        return FocalLoss(
            alpha=hyperparameters['alpha'],
            gamma=hyperparameters['gamma'],
            iou_thresholds=hyperparameters['iou_thresholds'],
            device=self.device
        )

    def get_optimizer(self):
        """Returns the optimizer of the training.

        Stochastic Gradient Descent:
        https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer of the training.
                For the optimizer package see: https://pytorch.org/docs/stable/optim.html
        """
        hyperparameters = self.hyperparameters['optimizer']
        return torch.optim.SGD(
            self.model.parameters(),
            lr=hyperparameters['learning_rate'],
            momentum=hyperparameters['momentum'],
            weight_decay=hyperparameters['weight_decay']
        )

    def get_scheduler(self):
        """Get the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.
        """
        hyperparameters = self.hyperparameters['scheduler']
        return ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=hyperparameters['factor'],
            patience=hyperparameters['patience'],
            verbose=True,
            threshold=hyperparameters['threshold']
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

        weights = self.hyperparameters['criterion']['weights']
        classification_loss *= weights['classification']
        regression_loss *= weights['regression']

        loss = classification_loss + regression_loss

        # Log the classification and regression loss too:
        self.current_log['Class.'] = float(classification_loss)
        self.current_log['Regr.'] = float(regression_loss)

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
        params = self.hyperparameters['model']['evaluation']
        self.model.eval(threshold=params['threshold'], iou_threshold=params['iou_threshold'], loss=True)
