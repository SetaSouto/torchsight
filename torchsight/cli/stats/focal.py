"""Commands to interact and get some stats of the Focal Loss."""
import click


@click.command()
@click.option('--alpha', default=0.25, show_default=True)
@click.option('--gamma', default=2.0, show_default=True)
@click.option('-p', '--prob', default=0.5, show_default=True)
def focal(alpha, gamma, prob):
    """Compute the focal loss for the given params."""
    import torch

    result = -alpha * (1 - prob) ** gamma * torch.log(torch.Tensor([prob]))

    print('{:.7f}'.format(result.item()))
