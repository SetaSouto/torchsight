"""Module with an evaluator for the Flickr32 dataset."""
import torch
from torch.utils.data import DataLoader

from torchsight.datasets import Flickr32Dataset
from torchsight.evaluators.map.bounding_box import BoundingBox
from torchsight.evaluators.map.bounding_boxes import BoundingBoxes
from torchsight.evaluators.map.evaluator import Evaluator as MapEvaluator
from torchsight.evaluators.map.utils import BBFormat, BBType

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
        self.map_evaluator = MapEvaluator()
        self.map_boxes = BoundingBoxes()

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_params():
        """Get the base parameters for the evaluator."""
        return Evaluator.get_params().merge({
            'root': './datasets/flickr32',
            'file': './flickr32_predictions.csv',
            'dataset': 'test',
            'dataloader': {
                'num_workers': 8,
                'shuffle': False,
                'batch_size': 8
            },
            'iou_threshold': 0.5
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
                data (sequence): Sequence of tuples as (image, boxes, info).
                    The boxes are torch.Tensor with x1, y1, x2, y2, label and shape `(num of boxes, 5)`

            Returns:
                torch.Tensor: The images.
                    Shape:
                        (batch size, channels, height, width)
                list of torch.Tensor: with the ground truth bounding boxes for each image.
                    Shape of the tensors: (number of annotations, 5) with x1, y1, x2, y2 and the label
                    of the class.
                list of str: The filename of the each image.
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
            annotations = [a for _, a, *_ in data]

            return images, annotations, files

        hyperparameters = {**self.params['dataloader'], 'collate_fn': collate}
        return DataLoader(**hyperparameters, dataset=self.dataset)

    ###############################
    ###         METHODS         ###
    ###############################

    def update_map(self, boxes_list, detections_list, file_names):
        """Update the Mean Average Precision metric with the detections of the current batch.

        Arguments:
            boxes_list (list of torch.Tensor): with the real ground truth annotations for the images where
                each tensor has shape (number of annotations, 5) with the x1, y1, x2, y2 and label
                of the class.
            detections_list (list of torch.Tensor): with the detections for each image. Each tensor
                has shape `(number of detections, 6)` with the x1, y1, x2, y2, label and probability
                for each detection.
            file_names (list of str): with the name of the images.
        """
        if len(boxes_list) != len(detections_list):
            raise ValueError('Ground truth and detection lists length mismatch')

        for i, boxes in enumerate(boxes_list):
            file_name = file_names[i]
            for box in boxes:
                x1, y1, x2, y2, label = box
                self.map_boxes.addBoundingBox(BoundingBox(
                    imageName=file_name,
                    classId=int(label),
                    x=int(x1), y=int(y1), w=int(x2), h=int(y2),
                    format=BBFormat.XYX2Y2,
                    bbType=BBType.GroundTruth
                ))
            for detection in detections_list[i]:
                x1, y1, x2, y2, label, prob = detection
                self.map_boxes.addBoundingBox(BoundingBox(
                    imageName=file_name,
                    classId=int(label),
                    x=int(x1), y=int(y1), w=int(x2), h=int(y2),
                    format=BBFormat.XYX2Y2,
                    bbType=BBType.Detected,
                    classConfidence=float(prob)
                ))

    def predict_one(self, detections_list, files):
        """Make the predictions for the given images.

        It predict only a single brand for each image to follow the evaluation kit of the dataset.

        It assumes that the model make predictions and returns a list of tensors with shape:
        `(num bounding boxes, 6)`.
        For each prediction contains x1, y1, x2, y2, label, probability.

        So this method keep only the maximum annotation and generates the tuples.

        If your model does not follow this structure you can override this method.

        Arguments:
            detections_list (list of torch.Tensor): The batch of detections for the images.
                Must be a list with a tensor for each image with shape `(num boxes, 6)`.
            infos (list of dict): A list of the dicts generated by the dataset.
                See __getitem__ method in the dataste for more information.

        Returns:
            list of tuples: Each tuple contains the name of the brand and the probability of
                the prediction.
                If the prediction is that there is no logo in the image it returns None as brand.
        """
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

    def forward(self, images, boxes, infos):
        """Forward pass through the model.

        Make the predictions and add it to the predictions variable.

        Arguments:
            torch.Tensor: The images.
                Shape: (batch size, channels, height, width)
            torch.Tensor: the ground truth bounding boxes for each image.
                Shape: (batch size, max number of annotations per image, 5).
            list of str: The filename of the each image.
        """
        images = images.to(self.device)
        detections_list = self.model(images)
        self.update_map(boxes, detections_list, infos)
        self.predictions += self.predict_one(detections_list, infos)
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

        # Get the average precision for each class
        metrics_per_class = self.map_evaluator.GetPascalVOCMetrics(
            self.map_boxes, IOUThreshold=self.params.iou_threshold)
        num_classes = len(metrics_per_class)
        for metric in metrics_per_class:
            brand = self.dataset.label_to_class[metric['class']]
            average_precision = metric['AP']
            print(f'{brand.ljust(15)}: {average_precision:.5f}')
        mean_ap = sum(metric['AP'] for metric in metrics_per_class) / num_classes
        print(f'\nmAP: {mean_ap:.5f}')
