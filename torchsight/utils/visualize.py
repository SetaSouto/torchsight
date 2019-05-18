"""Visualize images and annotations."""
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize_boxes(image, boxes, label_to_name=None):
    """Visualize an image and its bounding boxes.

    Arguments:
        image (PIL Image or torch.Tensor or numpy array): The image to show.
        boxes (torch.Tensor or numpy array): The bounding boxes with shape
            `(num boxes, 5 or 6)` with the x1,y1,x2,y2 for the top-left corner and
            the bottom-right corner and the index of the label to identify the class
            of the object. Optionally you can provide a 6th value for the confidence
            or probability of the bounding box.
        label_to_name (dict, optional): A dict to map the label of the class to its
            name.
    """
    if torch.is_tensor(image):
        image = image.numpy().transpose(1, 2, 0)
    if torch.is_tensor(boxes):
        boxes = boxes.numpy()

    n_colors = 20
    colormap = plt.get_cmap('tab20')
    colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]

    _, axes = plt.subplots(1)

    for box in boxes:
        if box.shape[0] == 6:
            x, y, x2, y2, label, prob = box
            prob = ' {:.2f}'.format(prob)
        else:
            x, y, x2, y2, label = box
            prob = ''

        w, h = x2 - x, y2 - y
        label = int(label)

        color = colors[label % n_colors]
        axes.add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))

        name = label_to_name[label] if label_to_name is not None else label

        tag = '{}{}'.format(name, prob)
        plt.text(x, y, s=tag, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

    print('Bounding boxes:\n{}'.format(boxes))

    axes.imshow(image)
    plt.show()
