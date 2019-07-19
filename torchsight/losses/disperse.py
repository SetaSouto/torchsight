"""Module with a loss to disperse the direction of the classification vectors of a model."""
import torch


class DisperseLoss(torch.nn.Module):
    """A loss to disperse the direction of the classification vectors.

    If you have a module with a matrix with the weights that perform the classification task
    given a set of embeddings, you probably has a matrix with shape `(e, c)` where `e` is
    the embedding size and `c` is the number of classes. This loss forces that this
    classification vectors points to different locations by simply penalizing their similarity.
    """

    def forward(self, weights):
        """Forward pass to get the loss given the classification weights.

        Arguments:
            weights (torch.Tensor): with the weights of the classification vectors.
                Shape `(e, c)` where `e` is the embedding size and `c` the number of classes
                so the classification vector of the `i` class is `weights[:, i]`.

        Returns:
            torch.Tensor: with the loss for this weights.
        """
        similarity = torch.matmul(weights.permute(1, 0), weights)               # (c, c)
        norms = weights.norm(dim=0)                                             # (c,)
        # Multiply the norm of each class vs all the other classes
        norms = torch.matmul(norms.unsqueeze(dim=1), norms.unsqueeze(dim=0))    # (c, c)
        similarity /= norms

        # Set the diagonal as zero, because the vector with itself will be always similar
        num_classes = weights.shape[1]
        indices = torch.arange(num_classes)
        similarity[indices, indices] = 0

        # The mean will be the sum of the similarity matrix divided by ((c ** 2) - c)
        # that is the total number of elements in the matrix (c ** 2) minus the diagonal of elements
        # (c) multiplied by 2 because we have duplicated the values, similarity[i, j] = similarity[j, i]
        num_items = ((num_classes ** 2) - num_classes)

        return similarity.sum() / (2 * num_items)
