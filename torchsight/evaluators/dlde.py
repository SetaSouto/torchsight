"""Evaluators for the DLDENet models."""
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchsight.datasets import CocoDataset
from torchsight.models import DLDENet, DLDENetWithTrackedMeans
from torchsight.transforms.detection import Normalize, Resize, ToTensor
from torchsight.utils import merge_dicts

from .evaluator import Evaluator


class DLDENetEvaluator(Evaluator):
    """An evaluator for the DLDENet.

    It will evaluate the model computing the mAP over the coco valid dataset.
    """
    params = {'results': {'dir': './evaluations/dldenet/coco', 'file': 'val2017.json'},
              'dataset': {'root': './datasets/coco',
                          'validation': 'val2017',
                          'class_names': (),
                          # Try to load the classes names from the checkpoint file
                          'class_names_from_checkpoint': True},
              'dataloader': {'batch_size': 8,
                             'shuffle': False,
                             'num_workers': 8},
              'model': {'with_tracked_means': False,
                        'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5},
                        # As the tracked version was created when the trainer didn't save the hyperparameters
                        # we must provide them
                        'tracked': {'classes': 80,
                                    'resnet': 50,
                                    'features': {'pyramid': 256,
                                                 'regression': 256,
                                                 'classification': 256},
                                    'anchors': {'sizes': [32, 64, 128, 256, 512],
                                                'scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)],
                                                'ratios': [0.5, 1, 2]},
                                    'embedding_size': 256,
                                    'concentration': 15,
                                    'shift': 0.8}},
              # We don't have params from the transforms because we load the params from the checkpoint,
              # but we can edit this params writing them here
              'transforms': {}}

    def __init__(self, *args, **kwargs):
        """Initialize the evaluator.

        Set the initial list with the predictions.
        """
        self.predictions = []

        super().__init__(*args, **kwargs)

    ###############################
    ###         GETTERS         ###
    ###############################

    def get_transform(self):
        """Get the transformations to apply to the dataset.

        Returns:
            torchvision.transforms.Compose: A composition of the transformations to apply.
        """
        params = torch.load(self.checkpoint)['hyperparameters']['transforms']
        params = merge_dicts(params, self.params['transforms'], verbose=True)

        return transforms.Compose([Resize(**params['resize']),
                                   ToTensor(),
                                   Normalize(**params['normalize'])])

    def get_dataset(self):
        """Get the COCO dataset for the evaluation.

        Returns:
            torch.utils.data.Dataset: The dataset to use for the evaluation.
        """
        params = self.params['dataset']
        transform = self.get_transform()
        class_names = params['class_names']

        if params['class_names_from_checkpoint']:
            checkpoint = torch.load(self.checkpoint)
            if 'hyperparameters' in checkpoint:
                class_names = checkpoint['hyperparameters']['datasets']['class_names']
            else:
                print("Couldn't load the class_names from the checkpoint, it doesn't have the hyperparameters.")

        return CocoDataset(
            root=params['root'],
            dataset=params['validation'],
            classes_names=class_names,
            transform=transform)

    def get_dataloader(self):
        """Get the dataloader to use for the evaluation.

        Returns:
            torch.utils.data.Dataloader: The dataloader to use for the evaluation.
        """
        def collate(data):
            """Custom collate function to join the different images.

            It pads the images so all has the same size.

            Arguments:
                data (sequence): Sequence of tuples as (image, annotations, images' infos, *_).

            Returns:
                torch.Tensor: The images.
                    Shape: (batch size, channels, height, width)
            """
            images = [image for image, *_ in data]
            max_width = max([image.shape[-1] for image in images])
            max_height = max([image.shape[-2] for image in images])

            def pad_image(image):
                aux = torch.zeros((image.shape[0], max_height, max_width))
                aux[:, :image.shape[1], :image.shape[2]] = image
                return aux

            images = torch.stack([pad_image(image) for image, *_ in data], dim=0)
            infos = [info for _, _, info, *_ in data]

            return images, infos

        return DataLoader(dataset=self.dataset, collate_fn=collate, **self.params['dataloader'])

    def get_model(self):
        """Get the model to use to make the predictions.

        We can use the DLDENet with tracked means or the weighted version by changing
        the flag params['model']['with_tracked_means'].

        Returns:
            torch.nn.Module: The model to use to make the predictions over the data.
        """
        if self.params['model']['with_tracked_means']:
            params = {**self.params['model']['tracked'], 'device': self.device}
            state_dict = torch.load(self.checkpoint, map_location=self.device)['DLDENet']
            return DLDENetWithTrackedMeans(**params).load_state_dict(state_dict)

        return DLDENet.from_checkpoint(self.checkpoint, self.device)

    ###############################
    ###         METHODS         ###
    ###############################

    def eval(self):
        """Put the model in evaluation mode and set the threshold for the detection."""
        params = self.params['model']['evaluation']
        self.model.eval(threshold=params['threshold'], iou_threshold=params['iou_threshold'])

    @staticmethod
    def transform_boxes(boxes):
        """Transform the bounding boxes from x1, y1, x2, y2 to x, y, width, height.

        Arguments:
            boxes (torch.Tensor): A tensor with shape (n predictions, 4).

        Returns:
            torch.Tensor: The transformed bounding boxes.
                Shape: (n predictions, 4)
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w, h = x2 - x1, y2 - y1
        return torch.stack([x1, y1, w, h], dim=1)

    def forward(self, images, infos, *_):
        """Forward pass through the network.

        Here we make the predictions over the images.

        Arguments:
            images (torch.Tensor): The tensor with the batch of images where to make predictions.
            infos (list): A list with the info of each image.
        """
        # Get the list of tuples (boxes, classifications)
        # With shapes (n predictions, 4) and (n predictions, n classes)
        predictions = self.model(images.to(self.device))
        for i, (boxes, classifications) in enumerate(predictions):
            # Check if there are no detections
            if boxes.shape[0] == 0:
                continue

            image_id = infos[i]['id']
            scores, labels = classifications.max(dim=1)
            boxes = self.transform_boxes(boxes)
            for j, box in enumerate(boxes):
                score, label = scores[j], labels[j]
                try:
                    category_id = self.dataset.classes['ids'][int(label)]
                except KeyError:
                    # The model predicted a class that is not present in the dataset
                    continue
                self.predictions.append({'image_id': image_id,
                                         'category_id': category_id,
                                         'bbox': [float(point) for point in box],
                                         'score': float(score)})

    def evaluate_callback(self):
        """After the finish of the evaluation store the predictions in the results directory
        and use the pycocotools to compute the mAP.
        """
        result_dir = self.params['results']['dir']
        file = self.params['results']['file']
        file_path = os.path.join(result_dir, file)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(file_path, 'w') as file:
            file.write(json.dumps(self.predictions))

        self.dataset.compute_map(file_path)
