"""Module to provide methods to calculate Intersection over Union."""
import torch

def iou(boxes, others):
    """Calculates the Intersection over Union between the given boxes.

    Each box must have the 4 values as x1, y1 (top left corner), x2, y2 (bottom
    right corner).

    Arguments:
        boxes (torch.Tensor): First group of boxes.
            Shape:
                (number of boxes in group 1, 4)
        others (torch.Tensor): Second group of boxes.
            Shape:
                (number of boxes in group 2, 4)

    Returns:
        torch.Tensor: The IoU between all the boxes in group 1 versus group 2.
            Shape:
                (number of boxes in group 1, number of boxes in group 2)

    How this work?

    Let's say there are 'd' boxes in group 1 and 'b' boxes in group 2.

    The beautiful idea behind this function is to unsqueeze the group 1 (add one dimension) to change the
    shape from (d) to (d, 1) and with the boxes with shape (b) the operation broadcast
    to finally get shape (d, b).

    Broadcast:
    PyTorch: https://pytorch.org/docs/stable/notes/broadcasting.html
    Numpy: https://docs.scipy.org/doc/numpy-1.10.4/user/basics.broadcasting.html

    The broadcast always goes from the last dimension to the first, so we have (b) with (d, 1), that is
    to look the tensors as:

    (d, 1)
    (   b)
    ------
    (d, b)

    So the final result is a tensor with shape (d, b).

    How can we read this broadcasting?

    The tensor with shape (d, 1) has "d rows with 1 column", and the vector with shape (b) has "b columns",
    this is the correct way to read the shapes. That's why if you print a tensor with shape (b) you get
    a vector as a row. And if you print a tensor with shape (d, 1) is like a matrix with d rows
    and 1 column. Always we start reading the tensors from the last element of its shape and in the order
    of "columns -> rows -> channels -> batches".

    So the first tensor with shape (d, 1) to match the (d, b) shape must repeat its single element of each
    row to match the column size. Ex:
    [[0],       [[0, 0, 0],
     [1],  -->   [1, 1, 1],
     [2]]        [2, 2, 2]]

    And the second tensor with shape (b) to match the (d, b) shape must repeat the values for each column
    to match the d rows. Ex:
    [3, 4, 5, 6]  --> [[3, 4, 5, 6],
                       [3, 4, 5, 6],
                       [3, 4, 5, 6]]

    Keep in mind this trick.
    Check how these "vectors" broadcast to the (d, b) matrix but with different ways. One repeat the column
    and the other repeat the rows.

    And now we can make calculations all vs all between the d and the b boxes without
    the need of a for loop. That's beautiful and efficient.

    Also, notice that the images are (channels, height, width), totally consequent with this vision on how
    to read the shapes.
    """
    # Shape (number of boxes)
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Shape (number of others)
    others_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])

    # Shape (number of boxes, number of others)
    intersection_x1 = torch.max(boxes[:, 0].unsqueeze(dim=1), others[:, 0])
    intersection_y1 = torch.max(boxes[:, 1].unsqueeze(dim=1), others[:, 1])
    intersection_x2 = torch.min(boxes[:, 2].unsqueeze(dim=1), others[:, 2])
    intersection_y2 = torch.min(boxes[:, 3].unsqueeze(dim=1), others[:, 3])

    intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)
    intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)
    intersection_area = intersection_width * intersection_height

    union_area = torch.unsqueeze(boxes_areas, dim=1) + others_areas - intersection_area
    union_area = torch.clamp(union_area, min=1e-8)

    return intersection_area / union_area
