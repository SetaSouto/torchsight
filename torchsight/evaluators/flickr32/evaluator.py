"""Module with an evaluator for the Flickr32 dataset."""
import torch
from torch.utils.data import DataLoader

from torchsight.datasets import Flickr32Dataset
from torchsight.utils import merge_dicts

from ..evaluator import Evaluator
from .fl_eval_classification import fl_eval_classification


class Flickr32Evaluator(Evaluator):
    """An evaluator for the Flickr32 dataset.

    You must extend this evaluator and override the `get_model()` method to use a custom model
    to perform the evaluation and `get_transform()` to use the transformation of the images
    as your model needs.

    The model should return the name of the brand for each image (or none if the image has
    no logo) and the probability of that prediction.
    You could override the method `predict()` to perform that task.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the evaluator."""
        self.processed = 0  # Number of processed images
        self.detected = 0  # Number of images with logos detected
        self.predictions = []

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_base_params():
        """Get the base parameters for the evaluator."""
        return merge_dicts(
            Evaluator.get_base_params(),
            {
                'root': './datasets/flickr32',
                'file': './flickr32_predictions.csv',
                'dataset': 'test',
                'dataloader': {
                    'num_workers': 8,
                    'shuffle': False,
                    'batch_size': 8
                }
            })

    ###############################
    ###         GETTERS         ###
    ###############################

    def get_transform(self):
        """Get the transformation to applies to the dataset according to the model."""
        raise NotImplementedError()

    def get_model(self):
        """Get the model that makes the predictions."""
        raise NotImplementedError()

    def get_dataset(self):
        """Get the dataset for the evaluation.

        Returns:
            torch.utils.data.Dataset: The dataset to use for the evaluation.
        """
        transform = self.get_transform()

        try:
            params = self.checkpoint['hyperparameters']['datasets']['flickr32']
        except KeyError:
            # The model was not trained over flickr32 dataset
            params = {'brands': None}

        return Flickr32Dataset(root=self.params['root'], brands=params['brands'], only_boxes=False,
                               dataset=self.params['dataset'], transform=transform)

    def get_dataloader(self):
        """Generate the custom dataloaders for the evaluation.

        Returns:
            torch.utils.data.Dataloaders: The dataloader for the validation.
        """
        def collate(data):
            """Custom collate function to join the images and get the name of the images.

            Arguments:
                data (sequence): Sequence of tuples as (image, _, info).

            Returns:
                torch.Tensor: The images.
                    Shape:
                        (batch size, channels, height, width)
                list of dicts: The filename of the each image.
            """
            images = [image for image, *_ in data]
            max_width = max([image.shape[-1] for image in images])
            max_height = max([image.shape[-2] for image in images])

            def pad_image(image):
                aux = torch.zeros((image.shape[0], max_height, max_width))
                aux[:, :image.shape[1], :image.shape[2]] = image
                return aux

            images = torch.stack([pad_image(image) for image, *_ in data], dim=0)
            files = [info['image'].split('/')[-1].replace('.jpg', '') for _, _, info in data]

            return images, files

        hyperparameters = {**self.params['dataloader'], 'collate_fn': collate}
        return DataLoader(**hyperparameters, dataset=self.dataset)

    ###############################
    ###         METHODS         ###
    ###############################

    def predict(self, images, files):
        """Make a predictions for the given images.

        It assumes that the model make predictions and returns a list of tensors with shape:
        `(num bounding boxes, 6)`.
        For each prediction contains x1, y1, x2, y2, label, probability.

        So this method keep only the maximum annotation and generates the tuples.

        If your model does not follow this structure you can override this method.

        Arguments:
            images (torch.Tensor): The batch of images to make predictions on.
            infos (list of dict): A list of the dicts generated by the dataset.
                See __getitem__ method in the dataste for more information.

        Returns:
            list of tuples: Each tuple contains the name of the brand and the probability of
                the prediction.
                If the prediction is that there is no logo in the image it returns None as brand.
        """
        detections_list = self.model(images)
        predictions = []

        for i, detections in enumerate(detections_list):
            self.processed += 1
            if detections.shape[0] > 0:
                self.detected += 1
                probs = detections[:, 5]
                prob, index = probs.max(dim=0)
                label = detections[index, 4]
                brand = self.dataset.label_to_class[int(label.long())]
            else:
                brand = 'no-logo'
                prob = 1.0

            predictions.append((files[i], brand, '{:.1f}'.format(float(prob))))

        return predictions

    def forward(self, images, infos):
        """Forward pass through the model.

        Make the predictions and add it to the predictions variable.
        """
        images = images.to(self.device)
        self.predictions += self.predict(images, infos)
        self.current_log['Processed'] = self.processed
        self.current_log['Detected'] = self.detected

    def evaluate_callback(self):
        """After all the predictions, use the evaluation kit to compute the metrics.

        The evaluation kit receives the root directory of the dataset and a CSV file
        with `\t` as separator with rows with `image - brand/no-logo - prob.
        So this method generates the CSV file and call the evaluation function.
        """
        with open(self.params['file'], 'w') as file:
            file.write('\n'.join(('\t'.join(prediction) for prediction in self.predictions)))

        fl_eval_classification(self.params['root'], self.params['file'], verbose=True)
