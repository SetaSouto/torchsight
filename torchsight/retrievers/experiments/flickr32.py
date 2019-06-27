"""A sanity check of the instance retriever.

We are going to look for an object of an image between 3 images (included itself!)
so any decent model must get the real object.
"""
import os

import torch
from PIL import Image

from torchsight.retrievers.resnet import ResnetRetriever


def main():
    boxes = torch.Tensor([[[269, 160, 168, 251]],
                          [[47, 53, 399, 249]]])
    print(boxes.shape)
    root = '/home/souto/datasets/flickr32/sanity_check/'
    images = [Image.open(os.path.join(root, image)) for image in ['apple.jpg', 'adidas.jpg']]

    retriever = ResnetRetriever(root=root)

    distances, boxes, paths, belongs_to = retriever.query(images, boxes, k=5, device='cpu')

    print(distances)
    print(boxes.shape)
    print(paths)
    print(belongs_to)


if __name__ == '__main__':
    main()
