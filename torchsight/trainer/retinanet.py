"""RetinaNet trainer."""
import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from ..datasets import CocoDataset
from ..transforms.detection import Normalize, Resize, ToTensor
from ..losses import FocalLoss
from ..models import RetinaNet
from .abstract import AbstractTrainer


class RetinaNetTrainer(AbstractTrainer):
    """Trainer for the RetinaNet model.

    For each one of the hyperparameters please visit the class docstring.
    """
    # Base hyperparameters, can be replaced in the initialization of the trainer:
    # >>> RetinaNetTrainer(hyperparameters={'RetinaNet': {'classes': 1}})
    hyperparameters = {
        'RetinaNet': {
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
            'pretrained': True
        },
        'FocalLoss': {
            'alpha': 0.25,
            'gamma': 2.0,
            'iou_thresholds': {
                'background': 0.4,
                'object': 0.5
            }
        },
        'datasets': {
            'root': '/media/souto/DATA/HDD/datasets/coco',
            'class_names': ('person')  # () indicates all classes
        },
        'dataloaders': {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 4
        },
        'optimizer': {
            'learning_rate': 1e-4,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        'transforms': {
            'resize': {
                'min_side': 608,
                'max_side': 960,
                'stride': 32
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }

    def __init__(self, hyperparameters={}, logs='./logs'):
        """Initialize the trainer.

        Arguments:
            hyperparameters (dict): The hyperparameters for the training.
            logs (str): Path to where store the logs. If None is provided it does not log the training.
        """
        super(RetinaNetTrainer, self).__init__(hyperparameters, logs)

    def get_model(self):
        """Initialize and get the RetinaNet.

        Returns:
            torch.nn.Module: The RetinaNet model.
        """
        hyperparameters = self.hyperparameters['RetinaNet']
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
        hyperparameters = self.hyperparameters['FocalLoss']
        return FocalLoss(
            alpha=hyperparameters['alpha'],
            gamma=hyperparameters['gamma'],
            iou_thresholds=hyperparameters['iou_thresholds']
        )

    def get_transform(self):
        """Initialize and get the transforms for the images.

        Returns:
            torchvision.transform.Compose: The Compose of the transformations.
        """
        hyperparameters = self.hyperparameters['transforms']
        return transforms.Compose([
            Resize(**hyperparameters['resize']),
            ToTensor(),
            Normalize(**hyperparameters['normalize'])
        ])

    def get_datasets(self):
        """Initialize and get the Coco dataset for training and evaluation.

        Returns:
            torch.utils.data.Dataset: The Coco dataset.
        """
        transform = self.get_transform()
        hyperparameters = self.hyperparameters['datasets']

        retina_classes = self.hyperparameters['RetinaNet']['classes']
        dataset_classes = len(hyperparameters['class_names'])
        if dataset_classes > 0 and retina_classes != dataset_classes:
            raise ValueError(' '.join(['The amount of classes for the RetinaNet model ({})'.format(retina_classes),
                                       'must be the same to the length of "class_names" in',
                                       'the "dataset" hyperparameters ({}).'.format(dataset_classes)]))

        return (
            CocoDataset(
                root=hyperparameters['root'],
                dataset='train2017',
                classes_names=hyperparameters['classes_names'],
                transform=transform
            ),
            CocoDataset(
                root=hyperparameters['root'],
                dataset='val2017',
                classes_names=hyperparameters['classes_names'],
                transform=transform
            )
        )

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
            to giv them the same shapes and stack them.
            Why -1?
            Because the FocalLoss could interpret that label and ingore it for the loss.

            Arguments:
                data (sequence): Sequence of tuples as (image, annotations).

            Returns:
                torch.Tensor: The images.
                    Shape:
                        (batch size, channels, height, width)
                torch.Tensor: The annotations.
                    Shape:
                        (batch size, biggest amount of annotations, 5)
            """
            images = torch.stack([image for image, _ in data], dim=0)
            max_annotations = max([annotations.shape[0] for _, annotations in data])

            def fill(annotations):
                aux = -1 * torch.ones((max_annotations, 5)).to(self.device)
                aux[:annotations.shape[0], :] = annotations
                return aux

            annotations = torch.stack([fill(a) for _, a in data], dim=0)
            return images, annotations

        hyperparameters = {**self.hyperparameters['dataloaders'], 'collate_fn': collate}
        return (
            DataLoader(dataset=self.dataset, *hyperparameters),
            DataLoader(dataset=self.valid_dataset, *hyperparameters)
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
