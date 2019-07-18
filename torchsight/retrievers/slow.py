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
            embeddings (torch.Tensor): with shape `(b, e, dim)`.

        Returns:
            torch.Tensor: with the distances with shape `(q, b, e)`.
        """
        queries = queries.unsqueeze(dim=1).unsqueeze(dim=2)  # (q, 1, 1, dim)
        embeddings = embeddings.unsqueeze(dim=0)  # (1, b, e, dim)

        return ((queries - embeddings) ** 2).sum(dim=3).sqrt()  # (q, b, e)

    @staticmethod
    def _cos_distance(queries, embeddings):
        """Compute the cosine distance between the queries and the embeddings.

        Arguments:
            queries (torch.Tensor): with shape `(q, dim)`.
            embeddings (torch.Tensor): with shape `(b, e, dim)`.

        Returns:
            torch.Tensor: with the distances with shape `(q, b, e)`.
        """
        queries_norm = queries.norm(dim=1).unsqueeze(dim=1).unsqueeze(dim=2)          # (q, 1, 1)
        embeddings_norm = embeddings.norm(dim=2).unsqueeze(dim=0)                     # (1, b, e)
        norms = queries_norm * embeddings_norm                                        # (q, b, e)
        queries = queries.permute(1, 0)                                               # (   d, q)
        # Now we con do the matmul with (b, e, d) x (d, q) = (b, e, q)
        similarity = torch.matmul(embeddings, queries)                                # (b, e, q)
        similarity = similarity.permute(2, 0, 1)                                      # (q, b, e)
        similarity /= norms

        return 1 - similarity

    def _search(self, queries, k):
        """Search in the dataset and get the tensor with the distances, bounding boxes and the paths
        of the images.

        Arguments:
            queries (torch.Tensor): the embeddings generated for each query object.
                Shape `(number of instances to search, embedding dim)`.
            k (int): The number of results to get.

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
        distances = 1e8 * queries.new_ones(num_queries, k)
        boxes = queries.new_zeros(num_queries, k, 4)
        paths = [[None for _ in range(k)] for _ in range(num_queries)]

        num_batches = len(self.dataloader)
        total_imgs = 0
        init = time.time()

        with torch.no_grad():
            self.model.to(self.device)
            for i, (images, batch_paths) in enumerate(self.dataloader):
                batch_size = images.shape[0]
                images = images.to(self.device)
                embeddings, batch_boxes = self.model(images)  # (b, e, d), (b, e, 4)
                actual_distances = self._distance(queries, embeddings)  # (q, b, e)

                for b in range(batch_size):
                    # Update the distances
                    distances = torch.cat([distances, actual_distances[:, b, :]], dim=1)  # (q, k + e)
                    distances, indices = distances.sort(dim=1)              # (q, k + e), (q, k + e)
                    distances, indices = distances[:, :k], indices[:, :k]   # (q, k), (q, k)
                    # Update the boxes
                    image_boxes = batch_boxes[b].unsqueeze(dim=0)           # (1, e, 4)
                    image_boxes = image_boxes.repeat(num_queries, 1, 1)     # (q, e, 4)
                    boxes = torch.cat([boxes, image_boxes], dim=1)          # (q, k + e, 4)
                    boxes = boxes[torch.arange(num_queries).unsqueeze(dim=1), indices, :]   # (q, k, 4)
                    # Update the paths: only the indices >= k are new paths
                    for q in range(num_queries):
                        for j in range(k):
                            if indices[q, j] >= k:
                                paths[q][j] = batch_paths[b]

                # Show some stats about the progress
                total_imgs += images.shape[0]
                self.logger.log({
                    'Batch': '{}/{}'.format(i + 1, num_batches),
                    'Time': '{:.3f} s'.format(time.time() - init),
                    'Images': total_imgs
                })

        return distances, boxes, paths
