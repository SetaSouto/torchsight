"""A module with an experiment for the retrievers using the Flickr32 dataset."""
import os
import random

import numpy as np
import torch
from PIL import Image

from torchsight.datasets import Flickr32Dataset
from torchsight.metrics.retrieval import AveragePrecision
from torchsight.retrievers.dldenet import DLDENetRetriever
from torchsight.retrievers.resnet import ResnetRetriever
from torchsight.utils import JsonObject, PrintMixin, visualize_boxes


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
            'distances_file': './distances.npy',
            'boxes_file': './boxes.npy',
            'retriever': {
                'use': 'dldenet',
                'dldenet': {
                    'extractor': 'fpn',
                    'checkpoint': None,
                    'paths': None,  # Populated by the experiment using the dataset's paths
                    'extensions': None,  # Not used
                    'batch_size': 8,
                    'num_workers': 8,
                    'instances_per_image': 1,
                    'verbose': True,
                    'device': None,  # Populated with the experiment's device
                    'transform_params': {}  # Populated with the checkpoint
                },
                'resnet': {
                    'paths': None,  # Populated by the experiment using the dataset's paths
                    'extensions': None,  # Not used
                    'batch_size': 8,
                    'num_workers': 8,
                    'instances_per_image': 1,
                    'verbose': True,
                    'device': None,  # Populated with the experiment's device
                    'params': {
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
                    }
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
        model = self.params.retriever.use
        params = self.params.retriever[model]
        params.device = self.device
        # Paths are tuples like (brand, image path, boxes path)
        params.paths = [t[1] for t in self.dataset.paths]

        if model == 'dldenet':
            if params.checkpoint is None:
                raise ValueError('Please provide a checkpoint for the retriever.')

            params.checkpoint = torch.load(params.checkpoint, map_location=self.device)
            params.transform_params = params.checkpoint['hyperparameters']['transform']

            return DLDENetRetriever(**params)

        if model == 'resnet':
            return ResnetRetriever(**params)

        raise NotImplementedError('There is no implementation for the "{}" retriever.'.format(model))

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
        distances, boxes, paths, _ = self.retriver.query(images=images, boxes=query_boxes, k=self.params.k)
        paths = self.unique_paths(paths)

        self.print('Getting average precision for each query ...')
        average_precisions = self.average_precision(*self.results_tensors(brands, paths))

        self.print('Storing results ...')
        self.store_results(distances, boxes, brands, paths, average_precisions)

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

    def store_results(self, distances, boxes, brands, paths, average_precisions):
        """Store the np.array results and a CSV file with columns: Brand, Average Precision, result paths.

        Arguments:
            distances (np.ndarray): The distances between the embedding queries and the found object in
                descendant order.
            boxes (np.ndarray): The bounding boxes for each result. Shape `(num of query objects, k, 4)`.
            brands (list of str): with the brand of each query.
            paths (list of list of str): with the paths of the images retrieved.
            average_precisions (torch.Tensor): with the average precision for each query.
        """
        np.save(self.params.distances_file, distances)
        np.save(self.params.boxes_file, boxes)

        with open(self.params.results_file, 'w') as file:
            for i, brand in enumerate(brands):
                line = '{},{:.5f},'.format(brand, float(average_precisions[i]))
                line += ','.join(paths[i])
                file.write(line + '\n')

    def visualize_results(self, brand=None, k=10):
        """Visualize the top k results of the given brand. If no brand is given it will iterate through
        each brand showing its top k results.

        Arguments:
            brand (str, optional): The brand to visualize.
            k (int, optional): the number of results to show.
        """
        distances = np.load(self.params.distances_file)[:, :k]  # (q, k)
        boxes = np.load(self.params.boxes_file)[:, :k, :]       # (q, k, 4)

        # Load the queries
        brands, images, query_boxes = self.load_queries()

        # Load the results' paths
        with open(self.params.results_file, 'r') as file:
            paths = {}
            for line in file.read().split('\n'):
                items = line.split(',')
                paths[items[0]] = items[2:(k+2)]

        # Filter by the brand
        if brand is not None:
            for i, b in enumerate(brands):
                if b == brand:
                    brands = [brand]
                    images = [images[i]]
                    query_boxes = [query_boxes[i]]
                    break

        # Show the results for each brand
        for i, brand in enumerate(brands):
            print('--------- {} --------'.format(brand))
            query_image = images[i]
            query_box = np.zeros(5)
            query_box[:4] = query_boxes[i]
            self.retriver.visualize(query_image, distances[i], boxes[i], paths[brand], query_box)

    def visualize_queries(self):
        """Visualize one by one the queries that the experiment is using."""
        brands, images, boxes = self.load_queries()

        for i, image in enumerate(images):
            box = torch.zeros(1, 5)     # (1, 5)
            box[:, :4] = boxes[i]       # (1, 5)
            visualize_boxes(image, box, label_to_name={0: brands[i]})
