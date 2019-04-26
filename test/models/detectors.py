"""Visualize some detections of the RetinaNet model."""
import argparse
import random

import torch
from torchvision import transforms

from torchsight.datasets import CocoDataset
from torchsight.models import DLDENet, RetinaNet
from torchsight.transforms.detection import Normalize, Resize, ToTensor

PARSER = argparse.ArgumentParser(
    description='Visualize some images from the CocoDataset with the predictions of the indicated model.')
PARSER.add_argument('model', help='The model to use. Options: "RetinaNet" and "DLDENet".')
PARSER.add_argument('root', help='The root directory where is the data.')
PARSER.add_argument('checkpoint', help='The checkpoint to load the model')
PARSER.add_argument('-d', '--dataset', nargs='?', default='val2017', help='The dataset to be loaded. Ex: "val2017"')
PARSER.add_argument('--no-random', action='store_const', const=True, default=False, help='Not show random images.')
PARSER.add_argument('-c', '--classes', nargs='+', default=(), help='The name of the classes that the model can detect')
PARSER.add_argument('-r', '--resnet', nargs='?', default=18, help='The ResNet backbone to use in the RetinaNet model.')
PARSER.add_argument('--threshold', default=0.5,
                    help='Keep only boxes with probability over this threshold. Default: 0.5')
PARSER.add_argument('--iou-threshold', default=0.2,
                    help='Set two boxes as the same if their IoU is over this threshold. Default: 0.2')

ARGUMENTS = PARSER.parse_args()

DATASET = CocoDataset(ARGUMENTS.root, ARGUMENTS.dataset, classes_names=ARGUMENTS.classes,
                      transform=transforms.Compose([Resize(), ToTensor(), Normalize()]))
# Not normalize the picture for human sight
DATASET_HUMAN = CocoDataset(ARGUMENTS.root, ARGUMENTS.dataset, classes_names=ARGUMENTS.classes,
                            transform=transforms.Compose([Resize(), ToTensor()]))
INDEXES = list(range(len(DATASET)))

N_CLASSES = len(ARGUMENTS.classes) if len(ARGUMENTS.classes) > 0 else 80

if ARGUMENTS.model.lower() == 'retinanet':
    MODEL = RetinaNet(classes=N_CLASSES, resnet=int(ARGUMENTS.resnet))
elif ARGUMENTS.model.lower() == 'dldenet':
    MODEL = DLDENet(classes=N_CLASSES, resnet=int(ARGUMENTS.resnet), embedding_size=256)
else:
    raise ValueError('There is no model with name {}'.format(ARGUMENTS.model))


# Set the device where to run
DEVICE = 'cpu'
CUDA = torch.cuda.is_available()
if CUDA:
    DEVICE = 'cuda:0'

# Load the checkpoint
MODEL.load_state_dict(torch.load(ARGUMENTS.checkpoint, map_location=DEVICE)['model'])
MODEL.to(DEVICE)

if not ARGUMENTS.no_random:
    random.shuffle(INDEXES)

MODEL.eval(threshold=float(ARGUMENTS.threshold), iou_threshold=float(ARGUMENTS.iou_threshold))

for index in INDEXES:
    image, ground = DATASET[index]
    image_human, _ = DATASET_HUMAN[index]
    # print('Ground truth:\n', ground)
    image = image.unsqueeze(0).type(torch.float)  # To simulate a batch
    if CUDA:
        image = image.to(DEVICE)
    boxes, classifications = MODEL(image)[0]  # Get the first result of the fake batch
    boxes, classifications = boxes[:100], classifications[:100]
    # print('Classifications:\n', classifications)
    if boxes.shape[0] == 0:
        DATASET.visualize(image_human)
        continue
    detections = torch.zeros((boxes.shape[0], 6))
    detections[:, :4] = boxes
    prob, label = classifications.max(dim=1)
    detections[:, 4] = label
    detections[:, 5] = prob
    # print('Detections:\n', detections)
    DATASET.visualize(image_human, detections.cpu())
