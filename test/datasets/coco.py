"""Visualize some images from the Coco dataset"""
import argparse
import random

from torchsight.datasets import CocoDataset
from torchsight.transforms.detection import Resize
from torchvision import transforms

PARSER = argparse.ArgumentParser(description='Visualize some images from the CocoDataset.')
PARSER.add_argument('root', help='The root directory where is the data.')
PARSER.add_argument('-d', '--dataset', nargs='?', default='val2017', help='The dataset to be loaded. Ex: "val2017"')
PARSER.add_argument('--no-random', action='store_const', const=False, default=True, help='Show random images.')

ARGUMENTS = PARSER.parse_args()

TRANSFORMS = transforms.Compose([Resize()])
DATASET = CocoDataset(ARGUMENTS.root, ARGUMENTS.dataset, classes_names=(), transform=TRANSFORMS)
INDEXES = list(range(len(DATASET)))

if ARGUMENTS.no_random:
    random.shuffle(INDEXES)

for index in INDEXES:
    DATASET.visualize_annotations(index)
