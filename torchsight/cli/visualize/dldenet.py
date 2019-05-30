"""Visualize the predictions of the dldenet."""
import random

import click
import torch
import torchvision

from torchsight.datasets import CocoDataset, Logo32plusDataset
from torchsight.models import DLDENet, DLDENetWithTrackedMeans
from torchsight.trainers import DLDENetTrainer
from torchsight.transforms.detection import Normalize
from torchsight.utils import visualize_boxes


@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
@click.argument('dataset-root', type=click.Path(exists=True))
@click.option('-d', '--dataset', default='coco', show_default=True, type=click.Choice(['coco', 'logo32plus']))
@click.option('--training-set', is_flag=True, help='Show the images of the training set instead of validation.')
@click.option('--no-shuffle', is_flag=True)
@click.option('--device', help='The device to use to run the model. Default to cuda:0 if cuda is available.')
@click.option('--threshold', default=0.5, show_default=True, help='The confidence threshold for the predictions.')
@click.option('--iou-threshold', default=0.3, show_default=True, help='The threshold for Non Maximum Supresion.')
@click.option('--tracked-means', is_flag=True)
def dldenet(checkpoint, dataset_root, dataset, training_set, no_shuffle, device, threshold, iou_threshold, tracked_means):
    """Visualize the predictions of the DLDENet model loaded from CHECKPOINT with the indicated
    dataset that contains its data in DATASET-ROOT."""
    device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint, map_location=device)

    if tracked_means:
        model = DLDENetWithTrackedMeans.from_checkpoint(checkpoint)
    else:
        model = DLDENet.from_checkpoint(checkpoint)

    hyperparameters = checkpoint['hyperparameters']

    transform = DLDENetTrainer.get_transform(hyperparameters['transforms'])
    transform_visible = torchvision.transforms.Compose(list(filter(lambda t: not isinstance(t, Normalize),
                                                                   [t for t in transform.transforms])))
    params = {'root': dataset_root}

    if dataset == 'coco':
        try:
            coco_params = hyperparameters['datasets']['coco']
        except KeyError:
            coco_params = hyperparameters['datasets']
        params['classes_names'] = coco_params['class_names']
        params['dataset'] = 'train2017' if training_set else 'val2017'
        dataset = CocoDataset(**params, transform=transform)
        dataset_human = CocoDataset(**params, transform=transform_visible)
        label_to_name = dataset.classes['names']
    elif dataset == 'logo32plus':
        params['dataset'] = 'training' if training_set else 'validation'
        dataset = Logo32plusDataset(**params, transform=transform)
        dataset_human = Logo32plusDataset(**params, transform=transform_visible)
        label_to_name = dataset.label_to_class
    else:
        raise ValueError('There is no dataset named "{}"'.format(dataset))

    indexes = list(range(len(dataset)))

    if not no_shuffle:
        random.shuffle(indexes)

    model.eval(threshold, iou_threshold)
    model.to(device)

    for i in indexes:
        image, *_ = dataset[i]
        image_visible, *_ = dataset_human[i]
        image = image.unsqueeze(dim=0).type(torch.float).to(device)
        boxes, classifications = model(image)[0]
        detections = torch.zeros((boxes.shape[0], 6))

        if boxes.shape[0]:
            detections[:, :4] = boxes
            prob, label = classifications.max(dim=1)
            detections[:, 4] = label
            detections[:, 5] = prob

        visualize_boxes(image_visible, detections, label_to_name)
