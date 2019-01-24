"""Module to compute mAP."""
import torch

from ..metrics import iou as compute_iou


class MeanAP():
    """Class to compute the Mean Average Precision.

    It follows the mAP fro COCO challenge:
    http://cocodataset.org/#detection-eval

    Inspired by Jonathan Hui from:
    https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    And Timothy C Arlen from:
    https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3
    """

    def __init__(self, start=0.5, stop=0.95, step=0.05):
        """Initialize the instance. Set the IoU thresholds.

        It evaluates AP from IoU@<start> to Iou@<stop> with a step between of <step>.
        Coco official metric is mAP@[.5:.05:.95].

        Arguments:
            start (float): The starting point for the IoU to calculate AP.
            stop (float): The final IoU to calculate AP.
            step (float): The step to advance from the 'start' to the 'stop'.
        """
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, annotations, detections):
        """Computes the mAP given the ground truth (annotations) and the detections.

        The annotations must have 5 values: x1, y1, x2, y2 (top left and bottom right corners
        of the bounding box) and the label of the class.

        The detections must have 6 values: The same 5 as ground truth but also a 6th value with the
        confidence of the detection (useful for sort the detections).

        Arguments:
            annotations (torch.Tensor): The ground truth.
                Shape:
                    (number of annotations, 5)
            detections (torch.Tensor): The detections from the model.
                Shape:
                    (number of detections, 6)

        Returns:
            torch.Tensor: mAP.
                Shape:
                    (1)
            torch.Tensor: All the AP for each IoU threshold.
                Shape:
                    (number of IoU thresholds between self.start and self.stop)
        """
        if annotations.shape[1] != 5:
            raise ValueError('Ground truth must have 5 values per annotation.')
        if detections.shape[1] != 6:
            raise ValueError('Detections must have 6 values per detection.')

        # Order detections by score
        _, order = detections[:, -1].sort(descending=True)
        detections = detections[order, :]

        # Compute Intersection over Union between the detections and the annotations
        iou = compute_iou(detections[:, :4], annotations[:, :4])  # (number of detections, number of annotations)

        # Get the assigned annotation for each detection by its max IoU with an annotation.
        # Now we can get the assigned annotation for each detection (for example, the detection 17 is assigned to the
        # ground truth annotation 9 with an IoU that is the maximum IoU of the detection with any annotation)
        iou_max, assigned_annotation = iou.max(dim=1)  # (number of detections)

        # Create a tensor that indicates with a 1 if the label of the detection correspond to its assigned annotation
        correct = annotations[assigned_annotation, -1] == detections[:, -2]  # Shape (number of detections)

        iou_thresholds = torch.range(self.start, self.stop, self.step)
        average_precisions = torch.zeros((iou_thresholds.shape[0]))

        for i, threshold in enumerate(iou_thresholds):
            # Keep only detections with an IoU with its assigned annotation over the threshold
            mask = iou_max >= threshold
            # Create the metrics tensor. It contains 3 metrics: Correct (0 or 1), Precision, Recall; per each detection
            # ordered by the confidence. So the recall must increase over the dimension 0 (the detections are ordered
            # by confidence at that dimension).
            n_current_detections = mask.sum()
            metrics = torch.zeros((n_current_detections, 3))
            metrics[:, 0] = correct[mask]
            # Get the number of expected correct labels
            n_total_annotations = annotations.shape[0]
            # Iterate over each detection and set precision and recall
            for j in range(n_current_detections):
                # Get the number of correct detections until now
                n_correct = metrics[:j + 1, 0].sum()
                # The number of total proposals until now
                n_proposals = j + 1
                # Compute precision and recall
                precision = n_correct / n_proposals
                recall = n_correct / n_total_annotations
                metrics[j, 1] = precision
                metrics[j, 2] = recall
            # Get the max precision over each recall between (0, 0.1, 0.2, ..., 1.0):
            precisions = torch.zeros((11))
            for j, recall in enumerate(torch.range(0, 1, 0.1)):
                # Generate the mask to keep only precisions over the current recall
                mask = metrics[:, 2] >= recall
                # Set the precision
                if mask.sum() > 0:
                    precisions[j] = metrics[mask, 1].max().item()
                else:
                    precisions[j] = 0.
            # Put the Average Precision
            average_precisions[i] = precisions.mean()
        # Return the mAP
        return average_precisions.mean(), average_precisions
