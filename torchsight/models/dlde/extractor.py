"""Module with the feature extractor version of the DLDENet."""
import torch

from .weighted import DLDENet


class DLDENetExtractor(DLDENet):
    """A model to get embeddings with bounding boxes instead or class proabilities."""

    def forward(self, images):
        """Generate the embeddings and bounding boxes for them for the given images.

        Arguments:
            images (torch.Tensor): of the batch with shape `(batch size, 3, height, width)`.

        Returns:
            torch.Tensor: with the embeddings. Shape `(batch size, num of embeddings, embedding dim)`.
            torch.Tensor: with the bounding boxes for the embeddings.
                Shape `(batch size, num of embeddings, 4)` with the `x1, y1, x2, y2` for the top-left
                corner and the bottom-right corner of the box.
        """
        with torch.no_grad():
            feature_maps = self.fpn(images)
            regressions = torch.cat([self.regression(feature_map) for feature_map in feature_maps], dim=1)
            embeddings = torch.cat([self.classification.encode(feature_map) for feature_map in feature_maps], dim=1)
            anchors = self.anchors(images)
            bounding_boxes = self.anchors.transform(anchors, regressions)
            bounding_boxes = self.anchors.clip(images, bounding_boxes)

            return embeddings, bounding_boxes


class FpnFromDLDENetExtractor(DLDENet):
    """A model that generates embeddings using the FPN of a DLDENet.

    It generates bounding boxes according to the size of the receptive field.

    For example, suppose an image with height = width = 512.
    The FPN has strides [8, 16, 32, 64, 128], so it generates feature maps
    with shape (b, c, s, s) with b = batchs size, c = channels, s = side, and the
    sides will be [64, 32, 16, 8, 4].
    You can note that the side also indicates how many receptive fields are in the
    feature map and the stride the size of the receptive field.
    We can now easily generate the bounding boxes for each feature map because
    the stride indicates the size of it.
    """

    def forward(self, images):
        """Generate the embeddings and bounding boxes for them for the given images.

        Arguments:
            images (torch.Tensor): of the batch with shape `(batch size, 3, height, width)`.

        Returns:
            torch.Tensor: with the embeddings. Shape `(batch size, num of embeddings, embedding dim)`.
            torch.Tensor: with the bounding boxes for the embeddings.
                Shape `(batch size, num of embeddings, 4)` with the `x1, y1, x2, y2` for the top-left
                corner and the bottom-right corner of the box.
        """
        with torch.no_grad():
            # list of (b, c, h, w) with different h, w
            feature_maps = self.fpn(images)
            # list of (b, h, w, 4) with different h, w
            boxes = self.get_boxes_from_feature_maps(feature_maps)
            batch_size, channels = feature_maps[0].shape[:2]
            # Transfrom the feature maps and bounding boxes to the expected output
            embeddings = torch.cat([f.view(batch_size, channels, -1) for f in feature_maps], dim=2)  # (b, c, e)
            embeddings = embeddings.permute(0, 2, 1).contiguous()                                    # (b, e, c)
            boxes = torch.cat([b.view(batch_size, -1, 4) for b in boxes], dim=1)                     # (b, e, 4)

            return embeddings, boxes

    def get_boxes_from_feature_maps(self, feature_maps):
        """Get the bounding boxes for the receptive fields in the feature maps.

        Arguments:
            feature_maps (list of torch.Tensor): with the features of the FPN where each one
                has shape `(batch size, num features, height, width)`.

        Returns:
            list of torch.Tensor: with the bounding boxes for each receptive field in each
                feature map. Each element in the list has shape `(batch size, height, width, 4)`
                with the `x1, y1` for the top left corner and the `x2, y2` for the bottom right
                corner.
        """
        boxes_list = []
        for s, feature_map in enumerate(feature_maps):
            stride = self.fpn.strides[s]
            batch_size, _, height, width = feature_map.shape
            boxes = feature_map.new_zeros(height, width, 4)
            for i in range(int(height)):
                for j in range(int(width)):
                    boxes[i, j, 0] = stride * i
                    boxes[i, j, 1] = stride * j
                    boxes[i, j, 2] = stride * (i+1)
                    boxes[i, j, 3] = stride * (j+1)
            boxes = boxes.unsqueeze(dim=0).repeat((batch_size, 1, 1, 1))
            boxes_list += [boxes]

        return boxes_list
