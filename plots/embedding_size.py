"""Script to plot how the size of the embedding affect the probabilities of the classes.
We generate big amounts of random embeddings and random classes' means, compute the dot product
and compute the maximums, minimums, averages and standard deviations to interpret how big is the
space to represent all the classes."""
import matplotlib.pyplot as plt
import torch


def main():
    sizes = range(1, 512)
    n_embedings = 1000
    n_means = 1000
    device = 'cuda'

    maxs = []
    mins = []
    averages = []
    stds = []

    for i, size in enumerate(sizes):
        print('{}/{}'.format(i + 1, len(sizes) + 1))
        embeddings = torch.rand((n_embedings, size)).to(device) * 2 - 1  # Between -1 and 1
        embeddings /= embeddings.norm(dim=1, keepdim=True)
        means = torch.rand((size, n_means)).to(device) * 2 - 1  # Between -1 and 1
        means /= means.norm(dim=0, keepdim=True)
        similarity = torch.matmul(embeddings, means)

        maxs.append(float(similarity.max()))
        mins.append(float(similarity.min()))
        averages.append(float(similarity.mean()))
        stds.append(float(similarity.std()))

    plt.subplot(2, 2, 1)
    plt.plot(sizes, maxs, '.-')
    plt.title('Maximums')

    plt.subplot(2, 2, 2)
    plt.plot(sizes, mins, '.-')
    plt.title('Mins')

    plt.subplot(2, 2, 3)
    plt.plot(sizes, averages, '.-')
    plt.title('Averages')

    plt.subplot(2, 2, 4)
    plt.plot(sizes, stds, '.-')
    plt.title('Stds')

    plt.show()


if __name__ == '__main__':
    main()
