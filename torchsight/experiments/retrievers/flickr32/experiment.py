"""A module with an experiment for the retrievers using the Flickr32 dataset."""
import os
import random

import torch
from PIL import Image

from torchsight.datasets import Flickr32Dataset
from torchsight.metrics.retrieval import AveragePrecision
from torchsight.retrievers.dldenet import DLDENetRetriever
from torchsight.utils import JsonObject, PrintMixin


class Flickr32RetrieverExperiment(PrintMixin):
    """An experiment to measure the precision, recall and F1 metrics using different retrivers over
    the Flickr32 dataset.

    The experiment consist that given some logo queries (images + bounding boxes) we need to retrieve
    all the instances of that logo. A perfect experiment will be to retrieve all the instances of all
    of the queries in the first place, without false positives.

    The experiment is not very strict, it takes a true positive if the image contains an instance, so
    if the retriever produced a very poor bounding box for the instance it does not matter. It focus
    on the finding of the instance.
    """

    def __init__(self, params=None, device=None):
        """Initialize the experiment.

        Arguments:
            params (dict, optional): a dict to modify the base parameters.
            device (str, optional): where to run the experiments. 
        """
        self.params = self.get_params().merge(params)
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.print('Loading dataset ...')
        self.dataset = self.get_dataset()
        self.print('Loading retriever ...')
        self.retriver = self.get_retriver()
        self.print('Loading metric ...')
        self.average_precision = AveragePrecision()

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
                'root': None,
                'dataset': 'test',
            },
            'k': 500,  # The amount of results to retrieve
            'queries_file': './queries.csv',
            'results_file': './results.csv',
            'retriever': {
                'use': 'dldenet',
                'dldenet': {
                    'checkpoint': None,
                    'paths': None,
                    'extensions': None,
                    'batch_size': 8,
                    'num_workers': 8,
                    'verbose': True,
                    'params': {'transform': {}}
                }
            }
        })

    def get_dataset(self):
        """Get the Flickr32 dataset.

        Returns:
            Flickr32Dataset: initialized and with its attributes.
        """
        return Flickr32Dataset(**self.params.dataset)

    def get_retriver(self):
        """Initialize and return the retriver to use in the experiment.

        Return:
            InstanceRetriver: the retriever to use.
        """
        retriever = self.params.retriever.use

        if retriever == 'dldenet':
            params = self.params.retriever.dldenet
            params.paths = [t[1] for t in self.dataset.paths]  # Paths are tuples like (brand, image path, boxes path)
            params.device = self.device

            if params.checkpoint is None:
                raise ValueError('Please provide a checkpoint for the DLDENet retriever.')

            params.checkpoint = torch.load(params.checkpoint, map_location=self.device)
            params.params.transform = params.checkpoint['hyperparameters']['transform']

            return DLDENetRetriever(**params)

        raise NotImplementedError('There is no implementation for the "{}" retriever.'.format(retriever))

    def generate_queries(self):
        """Generate a file with a random path to an image for each brand.

        It will store the file in the self.params.queries_file path.
        """
        results = []
        for brand in self.dataset.brands:
            if brand == 'no-logo':
                continue
            # Each path tuple contains (brand, image path, boxes path)
            results.append(random.choice([path for path in self.dataset.paths if path[0] == brand]))

        with open(self.params.queries_file, 'w') as file:
            file.write('\n'.join(','.join(line) for line in results))

    def load_queries(self):
        """Load the images and their bounding boxes to use as queries.

        It knows which images to use using the self.params.queries_file, if no one exists it generates a new one.

        Returns:
            list of str: with the brand of each image.
            list of PIL Image: with the images.
            list of torch.Tensor: with the first bounding box only to query. Shape `(1, 4)` with the
                x1, y1 for the top-left corner and the x2, y2 for the bottom-right corner.
        """
        if not os.path.exists(self.params.queries_file):
            self.generate_queries()

        brands, images, boxes = [], [], []
        with open(self.params.queries_file, 'r') as file:
            for brand, image, annot in [line.split(',') for line in file.read().split('\n')]:
                brands.append(brand)
                images.append(Image.open(image))
                with open(annot, 'r') as file:
                    line = file.readlines()[1]  # The first line contains "x y width height"
                    x, y, w, h = (int(val) for val in line.split())
                    x1, y1 = x - 1, y - 1
                    x2, y2 = x1 + w, y1 + h
                    boxes.append(torch.Tensor([[x1, y1, x2, y2]]))

        return brands, images, boxes

    ############################
    ###       METHODS        ###
    ############################

    def run(self):
        """Run the experiment and compute the mean average precision over the entire test dataset."""
        self.print('Loading queries ...')
        brands, images, query_boxes = self.load_queries()

        self.print('Retrieving instances ...')
        _, _, paths, _ = self.retriver.query(images=images, boxes=query_boxes, k=self.params.k)
        paths = self.unique_paths(paths)

        self.print('Getting average precision for each query ...')
        average_precisions = self.average_precision(*self.results_tensors(brands, paths))

        self.print('Storing results ...')
        self.store_results(brands, paths, average_precisions)

        self.print('Mean Average Precision: {}'.format(float(average_precisions.mean())))

    def unique_paths(self, paths):
        """Make the resulting paths unique.

        As each image could have more than one embedding similar to the query there could be images
        more than one time in the results, but we are want to count each image only one time.
        For this, we are going to remove the duplicates paths for each query.

        Arguments:
            paths (list of list of str): with the paths of the retrieved images.

        Returns:
            list of list of str: with the unique paths for each query.
        """
        for i in range(len(paths)):
            seen = set()
            paths[i] = [path for path in paths[i] if not (path in seen or seen.add(path))]

        return paths

    def results_tensors(self, brands, paths):
        """Generate the results tensor of the retrieval task for the metric and amount of relevant results.

        Arguments:
            brands (list of str): with the brand corresponding to each query.
            paths (list of list of str): with the resulting paths of the retrieved images for each query.

        Returns:
            torch.Tensor: with the 1s indicating the true positives. Shape `(q, k)`.
                Where `q` is the number of queries and `k` the number of instances retrieved.
            torch.Tensor: with the amount of relevant images for each query (true positives).
        """
        results = torch.zeros(len(brands), self.params.k)
        for i, brand in enumerate(brands):
            for j, path in enumerate(paths[i]):
                if brand in path:
                    results[i, j] = 1

        relevant = {brand: 0 for brand in brands}
        for brand, *_ in self.dataset.paths:
            if brand not in relevant:
                continue
            relevant[brand] += 1
        relevant = torch.Tensor(list(relevant.values()))

        return results, relevant

    def store_results(self, brands, paths, average_precisions):
        """Store the results in a CSV file with columns: Brand, Average Precision, result paths.

        Arguments:
            brands (list of str): with the brand of each query.
            paths (list of list of str): with the paths of the images retrieved.
            average_precisions (torch.Tensor): with the average precision for each query.
        """
        with open(self.params.results_file, 'w') as file:
            for i, brand in brands:
                line = '{},{},'.format(brand, float(average_precisions[i]))
                line += ','.join(paths[i])
                file.write(line + '\n')
