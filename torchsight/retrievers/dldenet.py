"""A module with a retriever based on the DLDENet."""
import torch

from torchsight.models.dlde.extractor import (DLDENetExtractor,
                                              FpnFromDLDENetExtractor)
from torchsight.transforms.augmentation import AugmentDetection
from torchsight.utils import JsonObject

from .slow import SlowInstanceRetriver


class DLDENetRetriever(SlowInstanceRetriver):
    """A retriever that uses the DLDENet extractor of the FPN extractor based on the weights
    of a DLDENet.
    """

    def __init__(self, checkpoint, *args, extractor='fpn', transform_params=None, device=None, **kwargs):
        """Initialize the retriver.

        Arguments:
            transform_params (dict, optional): The parameters for the transforms.

            The rest of the arguments are the same as the SlowInstanceRetriever, only the distance
            is fixed to 'cos' in the case of the DLDENet retriever or to 'l2' in the FpnFromDLDENet.
        """
        self.extractor = extractor
        self.transform_params = self.get_transform_params().merge(transform_params)
        self.checkpoint = checkpoint
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if extractor == 'fpn':
            distance = 'l2'
        elif extractor == 'dldenet':
            distance = 'cos'
        else:
            raise ValueError('The extractor must be "fpn" or "dldenet" not "{}"'.format(extractor))

        super().__init__(*args, **kwargs, distance=distance)

    @staticmethod
    def get_transform_params():
        """Get the base params for the transform.

        Returns:
            JsonObject: with the parameters for the transform.
        """
        return JsonObject({
            'LongestMaxSize': {
                'max_size': 512
            },
            'PadIfNeeded': {
                'min_height': 512,
                'min_width': 512
            }
        })

    def _get_model(self):
        """Get the correspondent detector."""
        if self.extractor == 'dldenet':
            return DLDENetExtractor.from_checkpoint(self.checkpoint, self.device)

        return FpnFromDLDENetExtractor.from_checkpoint(self.checkpoint, self.device)

    def _get_transforms(self):
        """Get the transformations to apply to the images in the dataset and in the queries.

        Returns:
            callable: a transformation for only images (the images where we are going to search).
            callable: a transformation for images and bounding boxes (the query images with their
                bounding boxes indicating the instances to search).
        """
        transform = AugmentDetection(self.transform_params, evaluation=True, normalize=True)

        return transform, transform
