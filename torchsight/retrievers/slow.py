"""Module with slow retrievers but with good memory footprint."""
import time

import torch

from .retriever import InstanceRetriever


class SlowInstanceRetriver(InstanceRetriever):
    """An implementation of an InstanceRetriever that to not abuse of the memory of the server
    it computes the embeddings for all the images every time a query is made.

    The algorithm is like:
    - Query K nearest instances for Q objects.
    - Generate the Q embeddings.
    - Iterate through the images by batching getting the nearest embeddings to the objects and
      update the k nearest ones.

    Returns the final k*Q nearest instances.
    """

    def __init__(self, *args, distance='l2', **kwargs):
        """Initialize the retriver.

        Arguments:
            distance (str, optional): The distance to use.

            The rest of the arguments are the same as InstanceRetriever.
        """
        if distance not in ['l2', 'cos']:
            raise ValueError('Distance "{}" not supported. Availables: {}'.format(distance, ['l2', 'cos']))

        self._distance = self._l2_distance if distance == 'l2' else self._cos_distance

        super().__init__(*args, **kwargs)

    @staticmethod
    def _l2_distance(queries, embeddings):
        """Compute the L2 distance between the queries and the embeddings.

        Arguments:
            queries (torch.Tensor): with shape `(q, dim)`.
            embeddings (torch.Tensor): with shape `(e, dim)`.

        Returns:
            torch.Tensor: with the distances with shape `(q, e)`.
        """
        queries = queries.unsqueeze(dim=1)  # (q, 1, dim)
        embeddings = embeddings.unsqueeze(dim=0)  # (1, e, dim)

        return ((queries - embeddings) ** 2).sum(dim=2).sqrt()  # (q, e)

    @staticmethod
    def _cos_distance(queries, embeddings):
        """Compute the cosine distance between the queries and the embeddings.

        Arguments:
            queries (torch.Tensor): with shape `(q, dim)`.
            embeddings (torch.Tensor): with shape `(e, dim)`.

        Returns:
            torch.Tensor: with the distances with shape `(q, e)`.
        """
        queries_norm = queries.norm(dim=1).unsqueeze(dim=1)                           # (q, 1)
        embeddings_norm = embeddings.norm(dim=1).unsqueeze(dim=0)                     # (1, e)
        norms = queries_norm * embeddings_norm                                        # (q, e)
        queries = queries.unsqueeze(dim=1).unsqueeze(dim=1)                           # (q, 1,   1, dim)
        embeddings = embeddings.unsqueeze(dim=0).unsqueeze(dim=3)                     # (1, e, dim,   1)
        similarity = torch.matmul(queries, embeddings).squeeze(dim=3).squeeze(dim=2)  # (q, e)

        return similarity / norms

    def _search(self, queries, k):
        """Search in the dataset and get the tensor with the distances, bounding boxes and the paths
        of the images.

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
        num_queries = queries.shape[0]
        distances = -1 * queries.new_ones(num_queries, k)
        boxes = queries.new_zeros(num_queries, k, 4)
        paths = [[None for _ in range(k)] for _ in range(num_queries)]

        num_batches = len(self.dataloader)
        total_imgs = 0
        init = time.time()

        with torch.no_grad():
            for i, (images, paths) in enumerate(self.dataloader):
                batch_size = images.shape[0]
                embeddings, boxes = self.model(images)
                actual_distances = self._distance(queries, embeddings)  # (q, e)

                for q, in range(num_queries):
                    for e in range(batch_size):
                        dis = actual_distances[q, e]
                        # Get the index of the first occurrence where the distance
                        # is bigger than the actual distance
                        index = (1 - (distances[q] > dis)).sum() - 1
                        if index >= k:
                            continue
                        distances[q, index] = dis
                        paths[q][index] = paths[e]
                        boxes[q, index, :] = boxes[e]

                # Show some stats about the progress
                total_imgs += images.shape[0]
                self.logger.log({
                    'Batch': '{}/{}'.format(i + 1, num_batches),
                    'Time': '{:.3f} s'.format(time.time() - init),
                    'Images': total_imgs
                })

        return distances, boxes, paths
