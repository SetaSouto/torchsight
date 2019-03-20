"""RetinaNet trainer."""
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            'pretrained': True,
            'evaluation': {
                'threshold': 0.5,
                'iou_threshold': 0.5
            }
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

    def __init__(self, *args, **kwargs):
        """Initialize the trainer."""
        self.compute_map = MeanAP()
        super(RetinaNetTrainer, self).__init__(*args, **kwargs)

    def forward(self, images, *args, **kwargs):
        """Forward pass through the network.

        Why this method? Is a software decision, that gives the freedom to override the forward pass
        through the network and reuse all the other code. See the DLDE trainer for an example.
        """
        return self.model(images)

    def train(self, epochs=100, validate=True, epoch_callback=None):
        """Train the model for the given epochs.

        Arguments:
            epochs (int): The number of epochs to train.
            validate (bool): If true it validates the model after each epoch using the validate method.
            epoch_callback (function, optional): An optional function to call after each epoch.
        """
        self.model.to(self.device)

        # Weights for each loss, to increase or decrease their values
        weights = self.hyperparameters['FocalLoss']['weights']

        print('----- Training started ------')
        print('Using device: {}'.format(self.device))

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
                anchors, regressions, classifications = self.forward(images, annotations)
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
                # Get the actual learning rate (modified by the scheduler)
                learning_rates = [str(param_group['lr']) for i, param_group in enumerate(self.optimizer.param_groups)]
                self.logger.log({
                    'Training': None,
                    'Epoch': epoch,
                    'Batch': batch_index,
                    'Time': '{:.3f}'.format(batch_time),
                    'Learning rate': ' '.join(learning_rates),
                    'Classification': '{:.7f}'.format(classification_loss),
                    'Regression': '{:.7f}'.format(regression_loss),
                    'Total': '{:.7f}'.format(loss)
                })

                # Save the weights for this epoch every some batches
                if batch_index % 100 == 0:
                    self.save_checkpoint(epoch)

            # Call the epoch callback
            if epoch_callback is not None:
                epoch_callback(epoch)

            # Save the weights at the end of the epoch
            self.save_checkpoint(epoch)

            if validate:
                validation_loss = self.validate(epoch)
                self.scheduler.step(validation_loss)

    def validate(self, epoch):
        """Compute the loss over the validation dataset."""
        self.model.to(self.device)
        hyperparameters = self.hyperparameters['RetinaNet']['evaluation']
        self.model.eval(threshold=hyperparameters['threshold'],
                        iou_threshold=hyperparameters['iou_threshold'],
                        loss=True)

        weights = self.hyperparameters['FocalLoss']['weights']

        classification_losses = []
        regression_losses = []
        losses = []

        start_time = time.time()
        last_endtime = time.time()

        for batch, (images, annotations) in enumerate(self.valid_dataloader):
            images, annotations = images.to(self.device), annotations.to(self.device)

            anchors, regressions, classifications = self.model(images)
            del images

            classification_loss, regression_loss = self.criterion(anchors, regressions, classifications, annotations)
            del anchors, regressions, classifications, annotations

            classification_loss *= weights['classification']
            regression_loss *= weights['regression']
            classification_loss = float(classification_loss)
            regression_loss = float(regression_loss)

            classification_losses.append(classification_loss)
            regression_losses.append(regression_loss)
            losses.append(classification_loss + regression_loss)

            batch_time = time.time() - last_endtime
            last_endtime = time.time()
            total_time = time.time() - start_time

            self.logger.log({
                'Validating': None,
                'Epoch': epoch,
                'Batch': batch,
                'Classification': '{:.7f}'.format(classification_loss),
                'Regression': '{:.7f}'.format(regression_loss),
                'Total': '{:.7f}'.format(losses[-1]),
                'Time': '{:.3f}'.format(batch_time),
                'Total time': '{:.3f}'.format(total_time)
            })

        return torch.Tensor(losses).mean()

    def validate_map(self):
        """Compute mAP over validation dataset.

        We iterate over the images in the validation dataset, compute the detections using the current
        state of the model, generate the detections tensor and compute the mAP using the MeanAP class.

        As the MeanAP class does not computes the mAP for each class (it computes for the entire image),
        we must filter the annotations by each class, compute mAP and store the value for the class and
        continue.

        Returns:
            torch.Tensor: The mAP averaged over all the classes.
        """
        print('--------- VALIDATING --------')

        self.model.to(self.device)
        self.model.eval(**self.hyperparameters['RetinaNet']['evaluation'])

        mAP = {}  # The array of mAP per each class per each image
        # mAP is something like {'0': [0.15, ..., 0.87]} where the length of the list is the number of images
        # in the validation dataset
        aps = {}  # The array with the Average Precision for each IoU threshold, for each image, for each class
        # aps is something like {'0': [[0.67, ..., 0.54], ..., []]} where the length of the bigger array is the
        # number of images that contains that label and the inner arrays is the number of IoU thresholds to compute
        # the Average Precisions

        for batch_index, (images, annotations) in enumerate(self.valid_dataloader):
            images, annotations = images.to(self.device), annotations.to(self.device)
            for index, (boxes, classifications) in enumerate(self.model(images)):
                # Generate the detections
                detections = torch.zeros((boxes.shape[0], 6)).to(self.device)
                detections[:, :4] = boxes
                prob, label = classifications.max(dim=1)
                detections[:, 4] = label
                detections[:, 5] = prob

                # Get the actual annotations, clean and iterate over each unique label
                actual_annotations = annotations[index].clone()
                # Remove dummy annotations created by the data loader (label == -1)
                mask = actual_annotations[:, -1] == -1
                actual_annotations = actual_annotations[mask]
                # Get the true labels in the actual annotation
                labels = [int(label) for label in actual_annotations[:, -1].unique()]
                # Iterate over each label to compute mAP per class
                for label in labels:
                    if label not in mAP:
                        mAP[label] = []
                    if label not in aps:
                        aps[label] = []
                    # Add zero values if there are no detections
                    if not boxes.shape[0] > 0:
                        mAP[label].append(torch.zeros((1)).mean().to(self.device))
                        aps[label].append(torch.zeros((self.compute_map.iou_thresholds.shape[0])).to(self.device))
                        continue
                    # Compute mAP
                    actual_map, actual_aps = self.compute_map(actual_annotations, detections)
                    mAP[label].append(actual_map)
                    aps[label].append(actual_aps)

            print('[Validating] [Batch {}]'.format(batch_index))

        # Set the model to train again
        self.model.train()
        # Compute the average of the map over all the classes
        final_map = [torch.stack(mAP[label]).mean() for label in mAP]
        final_aps = {label: torch.stack(aps[label]).mean(dim=0) for label in aps}
        return torch.stack(final_map).mean(), final_aps

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
            to give them the same shapes and stack them.
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
