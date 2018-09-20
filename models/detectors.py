"""Module that contains the Object Detector Model."""
import torch

from models.resnet import Resnet


class ObjectDetector(torch.nn.Module):
    """An object detector model heavily inspired by YOLO with a backbone of
    a Resnet.

    YOLO website:
    https://pjreddie.com/darknet/yolo/
    """

    def __init__(self, resnet_depth=50, anchors=((10, 13), (16, 30), (33, 23))):
        """Initialize the architecture.

        Set Resnet as feature extractor that has a network stride of 32 and an amount
        of output channels that depends on the depth of the network. It can be accessed
        by Resnet(resnet_depth).output_channels. For more information please see
        Resnet model.
        So, the resnet backbone has an output of size:
            (batch size, output channels, image size / 32, image size / 32).

        The prediction layer is a convolutional layer that takes the output channels
        and transforms them into the output of the network that predicts the bounding boxes
        and the confidence of them.
        It needs 5 parameters per bounding box: x, y, height, width and confidence.
        It uses the anchors as a reference and for each anchor it sets these 5 parameters as:
            bx = sigmoid(x) + cx
            by = sigmoid(y) + cy
            bh = anchor's height * exp(h)
            bw = anchor's width  * exp(w)
            confidence = sigmoid(confidence)
        Where cx and cy are the coordinates of the left top corner of the final grid.
        The sigmoid for x and y is to avoid centering a bounding box outside from it's grid
        cell. So an x and y of 0 mean that the center of the bounding box is at the top left
        corner and a x and y of 1 mean that the center of the bounding box is at the bottom
        right corner of the grid cell.

        So, for example, if you have an image of size (416, 416) it is reduced to an image of
        size (13, 13) (because the stride of 32) where for each cell of the 13x13 grid we
        predict the 5 parameters per anchor. So if you use 3 anchors it make 13x13x3
        predictions that contains 5 parameters, so the final size is as (making the channels
        the parameters for each anchor):
            (5*anchors, 13, 13)

        Anchors from the paper for an image of size 416x416:
        (10×13),(16×30),(33×23),(30×61),(62×45),(59×119),(116×90),(156×198),(373×326).

        Args:
            resnet_depth (int): Depth of the resnet backbone for feature extraction.
            anchors (tuple): Tuple of tuples representing the anchors to use to get the
                localization parameters.
        """
        self.anchors = anchors
        self.feature_extractor = Resnet(resnet_depth)
        # It predicts 5 values per anchor: (x, y, w, h, confidence)
        self.prediction_layer = torch.nn.Conv2d(in_channels=self.feature_extractor.output_channels,
                                                out_channels=5 * len(anchors),
                                                kernel_size=1)
        for module in self.prediction_layer:
            torch.nn.init.normal_(module.weight)

    def forward(self, inputs):
        """Forward pass of the network.

        Args:
            inputs (torch.Tensor): A tensor with shape as (batch size, 3, height, width).

        Returns:
            torch.Tensor: A tensor with shape (batch size, 5 * number of anchors,
                reduced size, reduced size) where reduced size is the image height and width
                reduced by a factor of 32. This is because the resnet has a stride of 32.
        """
        inputs = self.feature_extractor(inputs)
        inputs = self.prediction_layer(inputs)
        return self.format_output(input)


    def format_output(self, output):
        """Formats the output tensor to return a more friendly result.
        The output of the prediction layer is a tensor with size:
            (batch size, 5 * number of anchors, image reduced, image reduced)
        And this method returns a tensor as:
            (batch size, 5, number of anchors*image reduced*image reduced)
        So you can access each bounding box's parameter easily.

        Also it applies the sigmoids functions and adjust the predictions with the
        anchors and grid positions as mentioned at the __init__ method's documentation.

        Args:
            output (torch.Tensor): Output tensor after applying the feature extractor and the
                prediction layer.

        Returns:
            torch.Tensor: A tensor with the parameters of the bounding boxes adjusted with the
                anchors and positions at the final grid.
        """
        batch_size = output.shape[0]
        grid_size = output.shape[2]
        output = output.view(batch_size, 5, len(self.anchors), grid_size, grid_size)
        # Sigmoid x and y and adjust with the grid position
        grid_offsets = torch.arange(grid_size)
        output[:, 0, :, :, :] = (torch.sigmoid(output[:, 0, :, :, :]) + grid_offsets) / grid_size
        output[:, 1, :, :, :] = (torch.sigmoid(output[:, 1, :, :, :]) + grid_offsets) / grid_size
        # Apply anchor offsets
        for i, (anchor_height, anchor_width) in enumerate(self.anchors):
            output[:, 2, i, :, :] = anchor_height * torch.exp(output[:, 2, i, :, :])
            output[:, 3, i, :, :] = anchor_width * torch.exp(output[:, 3, i, :, :])
        # Apply sigmoid for confidence
        output[:, 4, :, :, :] = torch.sigmoid(output[:, 4, :, :, :])
        # Now, that each parameter was adjusted the grid position does not matter, so we can
        # return a tensor like (batch size, 5, number of predictions) or better a tensor
        # like (batch size, number of predictions, 5) where number of predictions is
        # grid size x grid size x number of anchors. And remember that grid size is the
        # image size reduced by 32 (the stride of the resnet backbone)
        output = output.view(batch_size, 5, len(self.anchors) * grid_size * grid_size)
        output = output.transpose(1, 2).contiguous()

        return output
