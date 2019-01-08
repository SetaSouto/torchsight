"""Visualize some images from the Coco dataset"""
import random
import sys
from os import path

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from datasets.coco import CocoDataset


# Configurations
ROOT = '/media/souto/DATA/HDD/datasets/coco'  # My custom root path
DATASET = 'val2017'  # The dataset that we are going to visualize
RANDOM = True  # Select random images

# Visualize
DATASET = CocoDataset(ROOT, DATASET)
INDEXES = list(range(len(DATASET)))

if RANDOM:
    random.shuffle(INDEXES)

for index in INDEXES:
    DATASET.visualize(index)
