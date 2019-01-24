"""Test the metricks package."""
import argparse
import sys
import unittest

import torch

from torchsight.datasets import CocoDataset
from torchsight.metrics import MeanAP

ROOT = None
DATASET = None


class TestMeanAP(unittest.TestCase):
    """Test the mAP metric."""

    def __init__(self, *args, **kwargs):
        """Initialize and set the dataset to obtain some annotations."""
        super(TestMeanAP, self).__init__(*args, **kwargs)
        self.dataset = CocoDataset(root=ROOT, dataset=DATASET)
        self.map = MeanAP()

    def get_annotations(self, image):
        """Get annotations for a given image.

        Arguments:
            image (int): The index of the image.

        Returns:
            torch.Tensor: Annotations for the image at the given index.
        """
        return torch.from_numpy(self.dataset[image][1]).type(torch.float)

    def get_detections_from(self, annotations):
        """Get a base detections tensor based on the ground truth annotations.

        Arguments:
            annotations (torch.Tensor): Base annotations.
                Shape:
                    (number of annotations, 5)

        Returns:
            detections (torch.Tensor): The detections tensor.
                Shape:
                    (number of annotations, 6)
        """
        detections = torch.zeros((annotations.shape[0], annotations.shape[1] + 1))
        detections[:, :-1] = annotations
        return detections

    def test_map_1(self):
        """Test that the map could be 1.

        It uses the same annotations as detections.
        """
        annotations = self.get_annotations(0)
        detections = self.get_detections_from(annotations)
        self.assertEqual(1., float(self.map(annotations, detections)[0]))

    def test_map_0(self):
        """Test that the map gives zero if there is no correct detection.

        It changes all the labels based on ground truth annotations.
        """
        annotations = self.get_annotations(1)
        detections = self.get_detections_from(annotations)
        detections[:, -2] = annotations[:, -1] + 1
        self.assertEqual(0., float(self.map(annotations, detections)[0]))

    def test_map_05(self):
        """Test that the map gives 0.5.

        It modifies the half of the annotations.
        """
        annotations = torch.Tensor([[250.8200,  168.2600,  320.9300,  233.1400,    0.0000],
                                    [435.3500,  294.2300,  448.8100,  302.0400,    2.0000],
                                    [447.4400,  293.9100,  459.6000,  301.5600,    2.0000],
                                    [460.5900,  291.7100,  473.3400,  300.1600,    2.0000],
                                    [407.0700,  287.2500,  419.7200,  297.1100,    2.0000],
                                    [618.0600,  289.3100,  629.6600,  297.2600,    2.0000],
                                    [512.3000,  294.0700,  533.4800,  299.6400,    2.0000],
                                    [285.5500,  370.5600,  297.6200,  389.7700,    0.0000],
                                    [61.6100,   43.7600,  107.8900,  122.0500,   33.0000],
                                    [238.5400,  158.4800,  299.7000,  213.8700,   37.0000]])
        detections = self.get_detections_from(annotations)
        half = detections.shape[0] // 2
        detections[half:, -2] = detections[half:, -2] + 1

        # As we set as incorrect the last 5 of the 10 annotations the precision will be:
        #    1,    1,    1,    1,    1,  5/6,  5/7,  5/8,  5/9, 5/10
        # And the recall will be:
        # 1/10, 2/10, 3/10, 4/10, 5/10, 5/10, 5/10, 5/10, 5/10, 5/10
        # So the average precision will be:
        # 1 + 1 + 1 + 1 + 1 + 1 + 0 + 0 + 0 + 0 + 0 = 6
        # So the AP = 6 / 11 = 0.545454

        self.assertEqual(0.54545, round(float(self.map(annotations, detections)[0]), 5))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Run the metrics tests.')
    PARSER.add_argument('root', help='The root directory to load the CocoDataset.')
    PARSER.add_argument('-d', '--dataset', nargs='?', default='val2017', help='The dataset to load. Ex: "val2017"')
    ARGUMENTS = PARSER.parse_args()
    ROOT = ARGUMENTS.root
    DATASET = ARGUMENTS.dataset
    # Clean arguments
    sys.argv = [command for command in sys.argv if command not in [ROOT, DATASET]]
    # Run the tests
    unittest.main()
