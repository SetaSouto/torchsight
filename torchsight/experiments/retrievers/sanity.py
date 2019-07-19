"""A sanity check of the instance retriever.

We are going to look for an object of an image between 3 images (included itself!)
so any decent model must get the real object.
"""
import os

import numpy as np
import torch
from PIL import Image

from torchsight.retrievers.dldenet import DLDENetRetriever
from torchsight.retrievers.resnet import ResnetRetriever


def main():
    query_boxes = torch.Tensor([[[269, 160, 168, 251]],
                                [[47, 53, 399, 249]]])  # (2, 1, 4)
    x1, y1, w, h = query_boxes[:, :, 0], query_boxes[:, :, 1], query_boxes[:, :, 2], query_boxes[:, :, 3]
    x2, y2 = x1 + w, y1 + h
    query_boxes = torch.stack([x1, y1, x2, y2], dim=2)

    root = '/home/souto/datasets/flickr32/sanity_check/'
    images = [Image.open(os.path.join(root, image)) for image in ['apple.jpg', 'adidas.jpg']]
    # retriever = ResnetRetriever(root=root, device='cpu')
    retriever = DLDENetRetriever(
        checkpoint="/home/souto/repos/pytorch/torchsight/logs/flickr32/resnet50/checkpoint.pth.tar", root=root)
    distances, boxes, paths, _ = retriever.query(images, query_boxes, k=5)

    for i, query_image in enumerate(images):
        query_image = retriever.image_transform({'image': query_image})
        box_with_dist = np.zeros(5)
        box_with_dist[:4] = query_boxes[i]
        retriever.visualize(query_image, distances[i], boxes[i], paths[i], query_box=box_with_dist)


if __name__ == '__main__':
    main()
