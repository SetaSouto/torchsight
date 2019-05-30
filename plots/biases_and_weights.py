"""Plot the power of the bias and the weight factor in a cosine similarity."""
import torch
from matplotlib import pyplot as plt


def main():
    step = 0.01
    similarities = torch.arange(start=-1, end=(1 + step), step=step)
    weights = torch.arange(start=0, end=25, step=1)
    biases = torch.arange(start=-0.8, end=0.4, step=0.4)
    sigmoid = torch.nn.Sigmoid()

    for w in weights:
        for b in biases:
            activations = sigmoid(w * (similarities + b))
            print(w, b)
            plt.scatter(similarities.numpy(), activations.numpy())
            plt.xlim((-1, 1))
            plt.ylim((0, 1))
            plt.show()


if __name__ == '__main__':
    main()
