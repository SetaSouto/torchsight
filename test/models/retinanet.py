"""Visualize some detections of the RetinaNet model."""
import argparse
import random

import torch

from torchsight.datasets import CocoDataset
from torchsight.transforms.detection import Resize, ToTensor, Normalize
from torchsight.models import RetinaNet
from torchvision import transforms

PARSER = argparse.ArgumentParser(
    description='Visualize some images from the CocoDataset with the predictions of the RetinaNet model.')
PARSER.add_argument('root', help='The root directory where is the data.')
PARSER.add_argument('-d', '--dataset', nargs='?', default='val2017', help='The dataset to be loaded. Ex: "val2017"')
PARSER.add_argument('--no-random', action='store_const', const=True, default=False, help='Not show random images.')
PARSER.add_argument('checkpoint', help='The checkpoint to load the model')
PARSER.add_argument('-c', '--classes', nargs='?', default=80, help='The number of classes that the model can detect')
PARSER.add_argument('-r', '--resnet', nargs='?', default=50, help='The ResNet backbone to use in the RetinaNet model.')


ARGUMENTS = PARSER.parse_args()

TRANSFORMS = transforms.Compose([Resize(), ToTensor(), Normalize()])
DATASET = CocoDataset(ARGUMENTS.root, ARGUMENTS.dataset, classes_names=(), transform=TRANSFORMS)
INDEXES = list(range(len(DATASET)))

MODEL = RetinaNet(classes=ARGUMENTS.classes, resnet=ARGUMENTS.resnet)
MODEL.load_state_dict(torch.load(ARGUMENTS.checkpoint)['model'])
CUDA = torch.cuda.is_available()
if CUDA:
    MODEL.to('cuda')

if not ARGUMENTS.no_random:
    random.shuffle(INDEXES)

MODEL.eval(threshold=0.4)

for index in INDEXES:
    image, ground = DATASET[index]
    print(ground)
    image = image.unsqueeze(0).type(torch.float)  # To simulate a batch
    if CUDA:
        image = image.to('cuda')
    boxes, classifications = MODEL(image)[0]  # Get the first result of the fake batch
    if boxes.shape[0] == 0:
        print('No detections')
        continue
    detections = torch.zeros((boxes.shape[0], 5))
    detections[:, :4] = boxes
    prob, label = classifications.max(dim=1)
    detections[:, 4] = label
    print(detections)
    print(classifications)
    DATASET.visualize(image[0].cpu(), detections.cpu())
