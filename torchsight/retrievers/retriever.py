"""Instance retriver."""
import math

import numpy as np
import torch
from PIL import Image

from torchsight.loggers import PrintLogger
from torchsight.metrics import iou as compute_iou
from torchsight.utils import visualize_boxes

from .datasets import ImagesDataset


class InstanceRetriever():
    """An abstract retriver that looks for instance of objects in a set of images."""

    def __init__(self, root=None, paths=None, extensions=None, batch_size=8, num_workers=8, verbose=True, device=None):
        """Initialize the retriever.

        You must provide the root directory of the images where to search of the paths of them.

        Arguments:
            root (str): The path to the root directory that contains the images
                where we want to search.
            paths (list of str): The paths of the images where to look for.
            extensions (list of str): If given it will load only files with the
                given extensions.
            batch_size (int, optional): The batch_size to use when processing the images with the model.
            num_workers (inr, optional): The number of workers to use to load the images and generate
                the batches.
            verbose (bool, optional): If True it will print some info messages while processing.
            device (str, optional): the device where to run the model. Default to cuda:0 if cuda is available.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._print('Loading model ...')
        self.model = self._get_model()
        self._print('Generating dataset ...')
        # Tuple with transforms: The first is only for images, the second images + boxes
        self.image_transform, self.with_boxes_transform = self._get_transforms()
        self.dataset = ImagesDataset(root=root, paths=paths, extensions=extensions, transform=self.image_transform)
        self.dataloader = self.dataset.get_dataloader(batch_size, num_workers)
        self.logger = PrintLogger()

    def _print(self, msg):
        """Print a namespaced message with the class' name.

        Arguments:
            msg (str): The message to print.
        """
        if self.verbose:
            print('[{}] {}'.format(self.__class__.__name__, msg))

    #############################
    ###        GETTERS        ###
    #############################

    def _get_model(self):
        """Get the model to generate the embeddings and bounding boxes.

        The model must be a callable model (i.e. `self.model()` must work) and must return
        a tuple with the embeddings generated for the batch of images and their bounding boxes.
        Specifically:
        - torch.Tensor: with shape `(batch size, num of embeddings, embedding dimension)`
        - torch.Tensor: with shape `(batch size, num of embeddings, 4)` with the `x1, y1, x2, y2`
            values for the top-left and bottom-right corners of the bounding box.

        Returns:
            callable: A model to generate the embeddings for the images.
        """
        raise NotImplementedError()

    def _get_transforms(self):
        """Get the transformations to apply to the images in the dataset and in the queries.

        Returns:
            callable: a transformation for only images (the images where we are going to search).
            callable: a transformation for images and bounding boxes (the query images with their
                bounding boxes indicating the instances to search).
        """
        raise NotImplementedError()

    #############################
    ###         SEARCH        ###
    #############################

    def query(self, images, boxes=None, strategy='max_iou', k=100):
        """Make a query for the given images where are instances of objects indicated with the boxes argument.

        If None is given for an image or for all, the retriver will set the bounding box as the image size,
        indicating that the object it's the predominant in the image.

        Arguments:
            images (list of PIL Images or np.array): a list with the PIL Images for the query.
            boxes (list of np.array or torch.Tensor, optional): a list with the tensors denoting the bounding boxes of
                the objects to query for each image. Each tensor must have `(num of objects, 4)` with its
                x1, y1, x2, y2 for the top-left corner and the bottom-right corner for each one of the objects
                to query that is in the image. For example, if the image has only one object to query you must
                provide an np.array/torch.Tensor like [[x1, y1, x2, y2]].
            strategy (str, optional): The strategy to use. If 'max_iou' it will query with the embedding with bigger
                IoU that generates the model. If 'avg' it will create an embedding with the weighted average of
                the embeddings with IoU above 0.5.
            k (int, optional): The number of results to get for each one of the object.

        Returns:
            np.ndarray: The distances between the embedding queries and the found object in descendant order.
                So the nearest result to the embedding query `i` has distance `distance[i, 0]`, and so on.
                To get the distances between the `i` embedding and its `j` result you can do
                `distances[i, j]`.
                Shape `(num of query objects, k)`.
            np.ndarray: The bounding boxes for each result. Shape `(num of query objects, k, 4)`.
            list of list of str: A list with `len = len(images)` that contains the path for each
                one of the images where the object was found.
                If you want to know the path of the result object that is in the `k`-th position
                of the `i` embedding you can do `results_paths[i][k]`.
            list of int: the index of the image that the embedding belongs to. Is useful to know the
                image of that embedding. To know the image from where is the embedding `i` you
                can do `belongs_to[i]`.
        """
        images, boxes = self._query_transform(images, boxes)
        queries, belongs_to = self._query_embeddings(images, boxes, strategy)  # (num of queries, embedding dim)
        distances, boxes, results_paths = self._search(queries, k)             # (num of queries, k)

        if torch.is_tensor(distances):
            distances = distances.numpy()

        if torch.is_tensor(boxes):
            boxes = boxes.numpy()

        return distances, boxes, results_paths, belongs_to

    def _query_transform(self, images, boxes):
        """Transform the inputs of the queries.

        Arguments:
            images (list of PIL Images or np.array): a list with the PIL Images for the query.
            boxes (list of np.array or torch.Tensor, optional): a list with the tensors denoting the bounding boxes of
                the objects to query for each image. Each tensor must have `(num of objects, 4)` with its
                x1, y1, x2, y2 for the top-left corner and the bottom-right corner for each one of the objects
                to query that is in the image. For example, if the image has only one object to query you must
                provide an np.array/torch.Tensor like [[x1, y1, x2, y2]].

        Returns:
            list of torch.Tensor: The images transformed.
            list of torch.Tensor: The boxes transformed.
        """
        images = [np.array(img) for img in images]

        # If there is no bounding box for any image
        if boxes is None:
            boxes = []
            for image in images:
                height, width = image.shape[:2]
                boxes.append(np.array([[0, 0, height, width]]))

        # If there is some None bounding boxes
        for i, image_boxes in enumerate(boxes):
            if image_boxes is None:
                height, width = images[i].shape[:2]
                boxes[i] = np.array([[0, 0, height, width]])

        # Transform the items
        for i, image in enumerate(images):
            image_boxes = boxes[i]
            image, image_boxes = self.with_boxes_transform({'image': image, 'boxes': image_boxes})
            images[i] = image
            boxes[i] = image_boxes

        return images, boxes

    def _query_embeddings(self, images, boxes, strategy):
        """Generate the embeddings that will be used to search.

        Arguments:
            images (list of torch.Tensor): the list of transformed images.
            boxes (list of torch.Tensor): the list of transformed bounding boxes.

        Returns:
            torch.Tensor: the embeddings generated for each instance.
                Shape `(number of instances to search, embedding dim)`.
            list of int: The index of the image where the embedding belongs. It has length
                `number of instances to search`. Se you can get the image index of the `i`
                embedding by doing `belongs_to[i]`.
        """
        num_images = len(images)

        # Make that the images have the same shape
        max_width = max([image.shape[2] for image in images])
        max_height = max([image.shape[1] for image in images])

        def pad_image(image):
            aux = torch.zeros((image.shape[0], max_height, max_width))
            aux[:, :image.shape[1], :image.shape[2]] = image
            return aux

        images = torch.stack([pad_image(image) for image in images], dim=0)

        # Process the images with the model
        with torch.no_grad():
            self.model.to(self.device)
            if num_images <= self.batch_size:
                images = images.to(self.device)
                batch_embeddings, batch_pred_boxes = self.model(images)     # (num images, *, dim), (num images, *, 4)
            else:
                batches = math.ceil(num_images / self.batch_size)
                batch_embeddings, batch_pred_boxes = [], []
                for i in range(batches):
                    batch = images[i * self.batch_size: (i + 1) * self.batch_size]
                    embeddings, pred_boxes = self.model(batch)
                    batch_embeddings.append(embeddings)
                    batch_pred_boxes.append(batch_pred_boxes)
                batch_embeddings = torch.cat(batch_embeddings, dim=0)       # (num images, *, dim)
                batch_pred_boxes = torch.cat(batch_pred_boxes, dim=0)       # (num images, *, 4)

        # Get the correct embedding for each query object
        result = []
        belongs_to = []
        for i, embeddings in enumerate(batch_embeddings):
            pred_boxes = batch_pred_boxes[i]         # (n pred, 4)
            iou = compute_iou(boxes[i], pred_boxes)  # (n ground, n pred)

            if strategy == 'max_iou':
                _, iou_argmax = iou.max(dim=1)       # (n ground)
                for embedding in embeddings[iou_argmax]:
                    result.append(embedding)
                    belongs_to.append(i)
            else:
                raise NotImplementedError()

        return torch.stack(result, dim=0), belongs_to

    def _search(self, queries, k):
        """Search in the dataset and get the tensor with the distances, bounding boxes and the paths
        of the images.

        **IMPORTANT**:
        Keep in mind that the bounding boxes are for the transformed images, not fot the original images.
        So, if the transformation changes the size of the image the bounding boxes could not fit
        in the original image.

        Arguments:
            queries (torch.Tensor): the embeddings generated for each query object.
                Shape `(number of instances to search, embedding dim)`.

        Returns:
            np.ndarray: The distances between the embedding queries and the found object in descendant order.
                So the nearest result to the embedding query `i` has distance `distance[i, 0]`, and so on.
                To get the distances between the `i` embedding and its `j` result you can do
                `distances[i, j]`.
                Shape `(num of query objects, k)`.
            np.ndarray: The bounding boxes for each result. Shape `(num of query objects, k, 4)`.
            list of list of str: A list with `len = len(images)` that contains the path for each
                one of the images where the object was found.
                If you want to know the path of the result object that is in the `k`-th position
                of the `i` embedding you can do `results_paths[i][k]`.
        """
        raise NotImplementedError()

    def visualize(self, query_image, distances, boxes, paths, query_box=None):
        """Show the query image and its results.

        Arguments:
            query_image (PIL Image or str): the path or the image that generates the query.
            distances (np.ndarray): The result distances for the query object.
                Shape: `(num results)`.
            boxes (np.ndarray): The boxes for the result embeddings.
                Shape: `(num results, 4)`.
            paths (list of str): The path to the result images.
            query_box (np.ndarray, optional): the bounding box of the query object.
        """
        if isinstance(query_image, str):
            query_image = Image.open(query_image)

        if query_box is None:
            query_box = []

        print('Query:')
        visualize_boxes(query_image, query_box)

        print('Results:')
        num_results = distances.shape[0]
        boxes_with_dist = torch.zeros(num_results, 5)       # (n, 5)
        boxes_with_dist[:, :4] = torch.Tensor(boxes)        # (n, 4)
        boxes_with_dist[:, 4] = torch.Tensor(distances)     # (n,)
        boxes_with_dist = boxes_with_dist.unsqueeze(dim=1)  # (n, 1, 5)
        for i, path in enumerate(paths):
            image = Image.open(path)
            image_box = boxes_with_dist[i]
            image = self.image_transform({'image': image})
            visualize_boxes(image, image_box)
