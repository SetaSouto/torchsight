"""Some stats about the DLDENet."""
import click


@click.group()
def dlde():
    """Show stats about the DLDENet."""


@dlde.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True), help='A checkpoint generated by a trainer.')
@click.option('--device', help='The device to use to load the model and make the computations.')
def weights(checkpoint, device):
    """Watch the norm of the classification weights of each class and generate a 
    similarity matrix between the each class weights.
    """
    import torch
    from matplotlib import pyplot as plt
    from torchsight.models import DLDENet

    device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

    weights = DLDENet.from_checkpoint(checkpoint, device=device).classification.weights

    with torch.no_grad():
        norms = weights.norm(dim=0)

        plt.subplot(1, 2, 1)
        plt.bar(x=[i for i in range(norms.shape[0])],
                height=norms.numpy())
        plt.title('The norm the classification vector for each class in the DLDENet')

        # weights has shape (embedding size, num classes)
        similarity = torch.matmul(weights.permute(1, 0), weights)
        # similarity is a matrix with shape (classes, classes) but is not normalized
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                similarity[i, j] /= norms[i] * norms[j]

        plt.subplot(1, 2, 2)
        plt.imshow(similarity.numpy())
        plt.colorbar()
        plt.title('Similarity matrix between the classification vectors in the DLDENet')
        plt.show()
