"""Visualize a modified sigmoid with weight and bias."""
import click


@click.command()
@click.option('-w', default=1., show_default=True)
@click.option('-b', default=0., show_default=True)
@click.option('--sigmoid-step', default=0.01)
@click.option('--normal', is_flag=True, help='Do not normalize for the maximum value of the cosine similarity.')
def sigmoid(w, b, sigmoid_step, normal):
    """Visualize a modified sigmoid with weight, bias and a maximum normalization."""
    import torch
    from matplotlib import pyplot as plt
    similarities = torch.arange(start=-1, end=(1 + sigmoid_step), step=sigmoid_step)
    sigmoid = torch.nn.Sigmoid()
    activations = sigmoid(w * similarities + b)
    if not normal:
        activations /= sigmoid(w * (torch.Tensor([1]) + b))
    plt.scatter(similarities.numpy(), activations.numpy())
    plt.xlim((-1, 1))
    plt.ylim((0, 1))
    plt.show()
