"""RetinaNet trainer."""
import time

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from ..datasets import CocoDataset
from ..losses import FocalLoss
from ..metrics import MeanAP
from ..models import RetinaNet
from ..transforms.detection import Normalize, Resize, ToTensor
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
            },
            # Weight of each loss. See train method.
            'weights': {'classification': 1e5, 'regression': 1}
        },
        'datasets': {
            'root': './datasets/coco',
            'class_names': (),  # () indicates all classes
            'train': 'train2017',
            'validation': 'val2017'
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

    def __init__(self, *args, **kwargs):
        """Initialize the trainer."""
        self.compute_map = MeanAP()
        super(RetinaNetTrainer, self).__init__(*args, **kwargs)

    def train(self, epochs=100, validate=True):
        """Train the model for the given epochs.

        Arguments:
            epochs (int): The number of epochs to train.
            validate (bool): If true it validates the model after each epoch using the validate method.
        """
        self.model.to(self.device)

        # Weights for each loss, to increase or decrease their values
        weights = self.hyperparameters['FocalLoss']['weights']

        print('----- Training started ------')
        print('Using device: {}'.format(self.device))

        if self.logger:
            print('Logs can be found at {}'.format(self.logger.log_file))

        for epoch in range(epochs):
            epoch = epoch + 1 + self.checkpoint_epoch
            last_endtime = time.time()

            # Set model to train mode, useful for batch normalization or dropouts modules. For more info see:
            # https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
            self.model.train()

            for batch_index, (images, annotations, *_) in enumerate(self.dataloader):
                images, annotations = images.to(self.device), annotations.to(self.device)

                # Optimize
                self.optimizer.zero_grad()
                anchors, regressions, classifications = self.model(images)
                del images
                classification_loss, regression_loss = self.criterion(anchors, regressions, classifications,
                                                                      annotations)
                del anchors, regressions, classifications, annotations

                classification_loss *= weights['classification']
                regression_loss *= weights['regression']
                loss = classification_loss + regression_loss
                # Set as float to free memory
                classification_loss = float(classification_loss)
                regression_loss = float(regression_loss)
                # Optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Log the batch
                batch_time = time.time() - last_endtime
                last_endtime = time.time()
                self.logger.log({
                    'Epoch': epoch,
                    'Batch': batch_index,
                    'Time': '{:.3f}'.format(batch_time),
                    'Classification': '{:.7f}'.format(classification_loss),
                    'Regression': '{:.7f}'.format(regression_loss),
                    'Total': '{:.7f}'.format(loss)
                })

                # Save the weights for this epoch every some batches
                if batch_index % 100 == 0:
                    self.save_checkpoint(epoch)

            # Save the weights at the end of the epoch
            self.save_checkpoint(epoch)

            if validate:
                self.validate()

    def validate(self):
        """Compute mAP over validation dataset."""
        print('--------- VALIDATING --------')

        self.model.to(self.device)
        self.model.eval()

        mAP = []
        aps = []

        for batch_index, (images, annotations) in enumerate(self.valid_dataloader):
            images, annotations = images.to(self.device), annotations.to(self.device)
            for index, (boxes, classifications) in enumerate(self.model(images)):
                detections = torch.zeros((boxes.shape[0], 6)).to(self.device)

                if not boxes.shape[0] > 0:
                    mAP.append(torch.zeros((1)).mean().to(self.device))
                    aps.append(torch.zeros((self.compute_map.iou_thresholds.shape[0])).to(self.device))
                    continue

                detections[:, :4] = boxes
                prob, label = classifications.max(dim=1)
                detections[:, 4] = label
                detections[:, 5] = prob
                actual_map, actual_aps = self.compute_map(annotations[index], detections)
                mAP.append(actual_map)
                aps.append(actual_aps)
            print('[Validating] [Batch {}] [mAP {:.3f}] [APs {}]'.format(
                batch_index,
                torch.stack(mAP).mean().item(),
                ' '.join(['{:.3f}'.format(ap.item()) for ap in torch.stack(aps).mean(dim=0)])))

        self.model.train()

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
                dataset=hyperparameters['train'],
                classes_names=hyperparameters['class_names'],
                transform=transform
            ),
            CocoDataset(
                root=hyperparameters['root'],
                dataset=hyperparameters['validation'],
                classes_names=hyperparameters['class_names'],
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

            Also it pads the images so all has the same size.

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
            images = [image for image, _ in data]
            max_width = max([image.shape[-1] for image in images])
            max_height = max([image.shape[-2] for image in images])

            def pad_image(image):
                aux = torch.zeros((image.shape[0], max_height, max_width))
                aux[:, :image.shape[1], :image.shape[2]] = image
                return aux

            images = torch.stack([pad_image(image) for image, _ in data], dim=0)

            max_annotations = max([annotations.shape[0] for _, annotations in data])

            def fill_annotations(annotations):
                aux = torch.ones((max_annotations, 5))
                aux *= -1
                aux[:annotations.shape[0], :] = annotations
                return aux

            annotations = torch.stack([fill_annotations(a) for _, a in data], dim=0)
            return images, annotations

        hyperparameters = {**self.hyperparameters['dataloaders'], 'collate_fn': collate}

        return (
            DataLoader(dataset=self.dataset, **hyperparameters),
            DataLoader(dataset=self.valid_dataset, **hyperparameters)
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
