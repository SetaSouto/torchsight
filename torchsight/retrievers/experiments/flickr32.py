"""A module with an experiment for the retrievers using the Flickr32 dataset."""
import torch

from torchsight.datasets import Flickr32Dataset
from torchsight.transforms.augmentation import AugmentDetection
from torchsight.utils import JsonObject

from ..dldenet import DLDENetRetriever


class Flickr32RetrieverExperiment():
    """An experiment to measure the precision, recall and F1 metrics using different retrivers over
    the Flickr32 dataset."""

    def __init__(self, params=None, device=None):
        """Initialize the experiment.

        Arguments:
            params (dict, optional): a dict to modify the base parameters.
            device (str, optional): where to run the experiments. 
        """
        self.params = self.get_params().merge(params)
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._print('Loading dataset ...')
        self.dataset = self.get_dataset()
        self._print('Loading retriever ...')
        self.retriver = self.get_retriver()

    def _print(self, msg):
        """Print a namespaced message.

        Arguments:
            msg (str): The message to print.
        """
        print('[{}] {}'.format(self.__class__.__name__, msg))

    ###########################
    ###       GETTERS       ###
    ###########################

    @staticmethod
    def get_params():
        """Get the base parameters for the experiment.

        Returns:
            JsonObject: with the parameters.
        """
        return JsonObject({
            'dataset': {
                'root': None
            },
            'retriever': {
                'use': 'dldenet',
                'dldenet': {
                    'checkpoint': None,
                    'root': None,
                    'paths': None,
                    'extensions': None,
                    'batch_size': 8,
                    'num_workers': 8,
                    'verbose': True,
                    'params': {
                        'transform': {
                            'LongestMaxSize': {
                                'max_size': 512
                            },
                            'PadIfNeeded': {
                                'min_height': 512,
                                'min_width': 512
                            }
                        }
                    }
                }
            },
            'transform': {}
        })

    def get_retriver(self):
        """Initialize and return the retriver to use in the experiment.

        Return:
            InstanceRetriver: the retriever to use.
        """
        retriever = self.params.retriever.use

        if retriever == 'dldenet':
            params = self.params.retriever.dldenet

            if params.checkpoint is None:
                raise ValueError('Please provide a checkpoint for the DLDENet retriever.')
            if params.root is None:
                raise ValueError('Please provide a root directory to scan for images.')

            params.checkpoint = torch.load(params.checkpoint, map_location=self.device)
            self.params.transform = params.checkpoint['hyperparameters']['transform']

            return DLDENetRetriever(**params)

        raise NotImplementedError('There is no implementation for the "{}" retriever.'.format(retriever))

    def get_dataset(self):
        """Get the Flickr32 dataset.

        Returns:
            Flickr32Dataset: initialized and with its attributes.
        """
        transform = AugmentDetection(params=self.params.transform, evaluation=True)
        return Flickr32Dataset(**self.params.dataset, dataset='test', transform=transform)

    ############################
    ###       METHODS        ###
    ############################

    def run(self):
        """Run the experiment and compute the mean average precision over the entire test dataset.

        # TODO:
        - Load all the images that contains a logo and keep the first bounding box as the query.
        - Get the paths of the retrieved images.
        - Generate the results tensor for the average metric.
        """
        raise NotImplementedError()
