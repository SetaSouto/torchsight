"""A sanity check of the instance retriever.

We are going to look for an object of an image between 3 images (included itself!)
so any decent model must get the real object.
"""
import os

import torch
from PIL import Image

from torchsight.retrievers.resnet import ResnetRetriever


def main():
    query_boxes = torch.Tensor([[[269, 160, 168, 251]],
                                [[47, 53, 399, 249]]])  # (2, 1, 4)
    for i, image_boxes in enumerate(query_boxes):
        for j, box in enumerate(image_boxes):
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            query_boxes[i, j] = torch.stack([x1, y1, x2, y2])

    root = '/home/souto/datasets/flickr32/sanity_check/'
    images = [Image.open(os.path.join(root, image)) for image in ['apple.jpg', 'adidas.jpg']]
    retriever = ResnetRetriever(root=root)
    distances, boxes, paths, _ = retriever.query(images, query_boxes, k=5, device='cpu')

    for i, query_image in enumerate(images):
        query_image = retriever.image_transform(query_image)
        box_with_dist = torch.zeros((1, 5))
        box_with_dist[:, :4] = torch.Tensor(query_boxes[i])
        box_with_dist[:, 4] = 0
        retriever.visualize(query_image, distances[i], boxes[i], paths[i], query_box=box_with_dist)


if __name__ == '__main__':
    main()
