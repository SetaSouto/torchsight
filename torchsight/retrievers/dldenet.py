"""A module with a retriever based on the DLDENet."""
import torch

from torchsight.models.dlde.extractor import DLDENetExtractor
from torchsight.transforms.augmentation import AugmentDetection
from torchsight.utils import JsonObject

from .slow import SlowInstanceRetriver


class DLDENetRetriever(SlowInstanceRetriver):
    """A retriever that uses the DLDENet extractor."""

    def __init__(self, checkpoint, *args, params=None, device=None, **kwargs):
        """Initialize the retriver.

        Arguments:
            params (JsonObject or dict, optional): The parameters for the model and the transforms.

            The rest of the arguments are the same as the SlowInstanceRetriever, only the distance
            is fixed to 'cos'.
        """
        self.params = self.get_params().merge(params)
        self.checkpoint = checkpoint
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super().__init__(*args, **kwargs, distance='cos')

    @staticmethod
    def get_params():
        """Get the base params for the model.

        Returns:
            JsonObject: with the parameters for the model.
        """
        return JsonObject({
            'transform': {
                'LongestMaxSize': {
                    'max_size': 512
                },
                'PadIfNeeded': {
                    'min_height': 512,
                    'min_width': 512
                }
            }
        })

    def _get_model(self):
        """Get the ResnetDetector."""
        return DLDENetExtractor.from_checkpoint(self.checkpoint, self.device)

    def _get_transforms(self):
        """Get the transformations to apply to the images in the dataset and in the queries.

        Returns:
            callable: a transformation for only images (the images where we are going to search).
            callable: a transformation for images and bounding boxes (the query images with their
                bounding boxes indicating the instances to search).
        """
        transform = AugmentDetection(self.params.transform, evaluation=True, normalize=True)

        return transform, transform
