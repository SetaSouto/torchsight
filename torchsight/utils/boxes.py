"""Some utils to work with the bounding boxes."""


def describe_boxes(boxes):
    """Describe the shapes of the bounding boxes.

    It computes the min, max, mean, median of the height, width and area of the bounding boxes.

    Arguments:
        boxes (torch.Tensor): Tensor with shape `(num of boxes, 4+)` with the x1, y1, x2 and y2 values
            for the top-left corner and bottom-right corner.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    width = x2 - x1
    height = y2 - y1
    area = width * height

    for name, tensor in [('Width', width), ('Height', height), ('Area', area)]:
        print('{}:'.format(name))
        print('  - Min:    {:.3f}'.format(float(tensor.min())))
        print('  - Max:    {:.3f}'.format(float(tensor.max())))
        print('  - Mean:   {:.3f}'.format(float(tensor.mean())))
        print('  - Median: {:.3f}'.format(float(tensor.median())))
