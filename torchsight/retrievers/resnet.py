"""A module for Resnet retrievers."""
from torchsight.models import ResnetDetector
from torchsight.transforms.augmentation import AugmentDetection
from torchsight.utils import JsonObject

from .slow import SlowInstanceRetriver


class ResnetRetriever(SlowInstanceRetriver):
    """A retriever that uses the dummy Resnet object detector."""

    def __init__(self, *args, params=None, **kwargs):
        """Initialize the retriver.

        Arguments:
            params (JsonObject or dict, optional): The parameters for the model and the transforms.

            The rest of the arguments are the same as the SlowInstanceRetriever, only the distance
            is fixed to 'l2'.
        """
        self.params = self.get_params().merge(params)
        super().__init__(*args, **kwargs, distance='l2')

    @staticmethod
    def get_params():
        """Get the base params for the model.

        Returns:
            JsonObject: with the parameters for the model.
        """
        return JsonObject({
            'model': {
                'resnet': 18,
                'dim': 512,
                'pool': 'avg',
                'kernels': [2, 4, 8, 16]
            },
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
        return ResnetDetector(**self.params.model)

    def _get_transforms(self):
        """Get the transformations to apply to the images in the dataset and in the queries.

        Returns:
            callable: a transformation for only images (the images where we are going to search).
            callable: a transformation for images and bounding boxes (the query images with their
                bounding boxes indicating the instances to search).
        """
        transform = AugmentDetection(self.params.transform, evaluation=True, normalize=True)

        return transform, transform
