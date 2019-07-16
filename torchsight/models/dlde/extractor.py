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
