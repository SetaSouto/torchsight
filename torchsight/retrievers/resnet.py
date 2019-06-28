"""A module for Resnet retrievers."""
from torchvision import transforms

from torchsight.models import ResnetDetector
from torchsight.transforms import detection
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
            'transforms': {
                'resize': {
                    'min_side': 384,
                    'max_side': 512,
                    'stride': 32
                },
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
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
        params = self.params.transforms

        image_transform = transforms.Compose([
            detection.Resize(**params.resize),
            transforms.ToTensor(),
            lambda x: x.float(),
            transforms.Normalize(**params.normalize)
        ])

        with_boxes_transform = transforms.Compose([
            detection.Resize(**params.resize),
            detection.ToTensor(),
            detection.Normalize(**params.normalize)
        ])

        return image_transform, with_boxes_transform