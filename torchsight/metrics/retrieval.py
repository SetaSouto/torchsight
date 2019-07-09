"""Module with metrics for retrievals tasks."""
import torch


class AveragePrecision():
    """Computes the average precision in a retrieval task.

    What's the difference with the mAP of an object detection task? Because here we use a different formula,
    we use the one indicated here:
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    """

    averages = []    # List of Average Precisions computed by this metric

    def __call__(self, results):
        """Compute the Average Precision for each query.

        The results tensor must have shape `(q, r)` where `q` is the number of queries
        and `r` is the number of results per query. The results must be labeled with a `1`
        when are correct (or relevant) and ordered in descendant order.

        Example:
        A tensor like:

        ```
        [[1., 1., 0., 1.],
         [0., 1., 0., 0.]]
        ```

        has `2` queries with `4` results. The first result for the first query is correct
        but the first result for the second query is incorrect. The first query has 3 correct
        results and the second query only one.

        Arguments:
            results (torch.Tensor): the ordered results of the queries labeled as correct
                with a 1.

        Returns:
            torch.Tensor: with the average precision of each query.
        """
        if len(results.shape) == 1:
            results = results.unsqueeze(dim=0)

        if len(results.shape) != 2:
            raise ValueError('"results" can only be a tensor of shape (q, r).')

        # Get the number of relevant results
        relevant = results.sum(dim=1)                              # (q, 1)

        if (relevant == 0).sum() >= 1:
            raise ValueError('There are queries without relevant results and could generate NaN results.')

        # Get the precision @k for each k in r
        precision = torch.zeros_like(results)                       # (q, r)
        for k in range(1, results.shape[1] + 1):
            precision[:, k - 1] = results[:, :k].sum(dim=1) / k   # (q,)

        # Set as zero the precision when the k-th element was not relevant
        precision[results != 1] = 0

        # Get the average precision for each query
        avg = precision.sum(dim=1) / relevant                      # (q,)

        self.averages.append(avg)

        return avg

    def mean(self):
        """Compute the mean average precision based on the past average precision computed
        stored in the self.averages list.

        Returns:
            torch.Tensor: with shape `(1,)` with the mean of the average precisions computed.
        """
        return torch.cat(self.averages).mean()
