import torch
import sys

from os.path import abspath, dirname, join
from unittest import TestCase

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add dataset module to path
data_path = abspath(join(dirname(abspath(__file__)), '../../datasets'))
sys.path.insert(0, data_path)

from coco.CocoDataset import CocoDataset


class COCOTest(TestCase):
    def setUp(self):
        # This is my custom path, change it to pass this test
        coco_path = '/media/souto/DATA/HDD/datasets/coco'
        self.dataset = CocoDataset(coco_path)  # train=True

    def test_length(self):
        self.assertEqual(len(self.dataset), 117264)

    def test_get_item(self):
        for i in range(10):
            image_path, image, bounding_boxes = self.dataset[i]
            self.assertTrue(isinstance(image_path, str))
            self.assertTrue(isinstance(image, torch.Tensor))
            self.assertEqual(len(image.shape), 3)
            self.assertEqual(image.shape[0], 3)
            self.assertTrue(isinstance(bounding_boxes, torch.Tensor))
            self.assertEqual(bounding_boxes.shape[1], 5)
            self.dataset.visualize_bounding_boxes(
                image=image, bounding_boxes=bounding_boxes)
            image = torch.from_numpy(np.array(Image.open(image_path)))
            bounding_boxes = torch.from_numpy(np.loadtxt(image_path.replace(
                'images', 'labels').replace('jpg', 'txt')).reshape(-1, 5))
            self.dataset.visualize_bounding_boxes(
                image=image, bounding_boxes=bounding_boxes)

    def test_gray_image(self):
        image_path, image, bounding_boxes = self.dataset[13]
        self.assertTrue(isinstance(image_path, str))
        self.assertTrue(isinstance(image, torch.Tensor))
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 3)
        self.assertTrue(isinstance(bounding_boxes, torch.Tensor))
        self.assertEqual(bounding_boxes.shape[1], 5)
        self.dataset.visualize_bounding_boxes(
            image=image, bounding_boxes=bounding_boxes)
        image = np.array(Image.open(image_path))
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)
        image = torch.from_numpy(image)
        bounding_boxes = torch.from_numpy(np.loadtxt(image_path.replace(
            'images', 'labels').replace('jpg', 'txt')).reshape(-1, 5))
        self.dataset.visualize_bounding_boxes(
            image=image, bounding_boxes=bounding_boxes)
