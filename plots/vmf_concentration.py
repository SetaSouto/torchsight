"""Script to visualize how important is the concentration factor for the pseudo probability function based
on the von Mises-Fisher distribution."""
import matplotlib.pyplot as plt
import torch


def main():
    """Plot the distribution using different concentration (K) values."""
    ks = [1] + list(range(5, 30, 5))
    step = 0.01
    x = torch.arange(-1, 1 + step, step)
    b = 0.8

    for i, k in enumerate(ks):
        y = 1 / (torch.exp(-k * (x - b)) + 1)
        n_rows = len(ks) // 2 + (1 if len(ks) % 2 != 0 else 0)
        plt.subplot(n_rows, 2, i + 1)
        plt.plot(x.numpy(), y.numpy(), '.-')
        plt.title('K: {}, Max: {:.2f}, Min: {:.2f}'.format(k, y.max(), y.min()))
        plt.ylim(0, 1)
        plt.subplots_adjust(hspace=0.4)

    plt.show()


if __name__ == '__main__':
    main()
