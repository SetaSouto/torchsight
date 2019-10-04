"""Mixins for the datasets."""
from torchsight.utils import visualize_boxes


class VisualizeMixin():
    """Add a method to visualize the bounding boxes."""

    def visualize(self, *args):
        """Visualize the annotations for the item in the given index.

        Arguments:
            index (int): The index of the item to visualize.

            Or

            image (torch.Tensor): The image to visualize.
            boxes (torch.Tensor): The bounding boxes of the given image.
        """
        if len(args) == 1 and isinstance(args[0], int):
            index, *_ = args
            image, boxes, *_ = self[index]
        elif len(args) == 2:
            image, boxes = args
        else:
            raise ValueError('Please provide inly the index or the only the image and the bounding boxes.')

        visualize_boxes(image, boxes, self.label_to_class)
