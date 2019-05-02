"""With this script we can visualize how the different norms of the classification vectors
could affect the decision over the hypersphere."""
import torch

import matplotlib.pyplot as plt


def plot_classification_weights(classes=5, perimeter_step=0.1):
    # First create the unit circle
    x = torch.arange(-1, 1 + perimeter_step, perimeter_step)
    y = torch.sqrt(1 - x ** 2)
    x = torch.cat((x, x))
    y = torch.cat((y, -y))

    embeddings = torch.stack((x, y)).t()  # Shape (number of points, 2)
    weights = torch.rand(2, classes) * 2 - 1
    cos_sim = torch.matmul(embeddings, weights)  # Shape (number of points, classes)

    max_sim, arg_max = cos_sim.max(dim=1)
    area = (max_sim * 10) ** 2

    plt.scatter(x.numpy(), y.numpy(), c=arg_max.numpy(), s=area.numpy())
    plt.scatter(weights[0].numpy(), weights[1].numpy(), c=torch.arange(classes).numpy())
    plt.scatter(torch.Tensor([0]).numpy(), torch.Tensor([0]).numpy(), color='b')
    plt.show()


if __name__ == '__main__':
    plot_classification_weights()
