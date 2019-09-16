"""Module with the evaluators of the RetinaNet model."""
from torchsight.models.retinanet import RetinaNet
from torchsight.transforms.augmentation import AugmentDetection
from torchsight.utils import merge_dicts

from .flickr32 import Flickr32Evaluator


class RetinaNetFlickr32Evaluator(Flickr32Evaluator):
    """Extend the Flickr32Evaluator class to perform an evaluation over the dataset using the evaluation kit
    provided with it.
    """

    @staticmethod
    def get_base_params():
        """Add the thresholds to the base parameters."""
        return merge_dicts(
            Flickr32Evaluator.get_base_params(),
            {
                'thresholds': {
                    'detection': 0.3,
                    'iou': 0.3
                }
            }
        )

    def eval_mode(self):
        """Put the model in evaluation mode and set the threshold for the detection."""
        params = self.params['thresholds']
        self.model.eval(threshold=params['detection'], iou_threshold=params['iou'])

    def get_transform(self):
        """Get the transformation to applies to the dataset according to the model."""
        return AugmentDetection(params=self.checkpoint['hyperparameters']['transform'], evaluation=True)

    def get_model(self):
        """Get the model to use to make the predictions.

        Returns:
            RetinaNet: The model loaded from the checkpoint.
        """
        return RetinaNet.from_checkpoint(self.checkpoint, self.device)
