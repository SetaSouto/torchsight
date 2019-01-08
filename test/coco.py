"""Visualize some images from the Coco dataset"""
import random
import sys
from os import path

from torchvision import transforms

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from datasets.coco import CocoDataset
from datasets.transforms import Resize


# Configurations
ROOT = '/media/souto/DATA/HDD/datasets/coco'  # My custom root path
DATASET = 'val2017'  # The dataset that we are going to visualize
RANDOM = True  # Select random images
TRANSFORMS = transforms.Compose([Resize()])

# Visualize
DATASET = CocoDataset(ROOT, DATASET, classes_names=(), transform=TRANSFORMS)
INDEXES = list(range(len(DATASET)))

if RANDOM:
    random.shuffle(INDEXES)

for index in INDEXES:
    DATASET.visualize(index)
