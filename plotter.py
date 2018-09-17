"""This module contains a class to use to lively plot the loss of the training."""
import matplotlib.pyplot as plt
import numpy as np


class LossPlotter:
    """A class to set a plotter of the training loss.
    Useful to see in real time how the train loss and the validation loss are
    improving or not.
    It loads the logs from the Trainer class and plot the loss using matplotlib's
    pyplot.
    """

    def __init__(self, logs_path, mean=True, window_size=100):
        """Initialize the plotter.
        It can plot the mean of the loss for a given window of batches.
        So if you provide a window's size of 100 it calculate the mean of the loss
        for 100 batches and add it to the plot.

        The log file must fit the scheme from the Trainer class. See ./trainer.py.

        Args:
            logs_path (str): The path to the logs file to read from.
            mean (bool): Indicates if you want to see the mean of the loss.
                Set to False if you want to see the loss for each batch.
            window_size (int): The size of the window of batches to calculate the mean
                and plot it.
        """
        self.logs_path = logs_path
        self.mean = mean
        self.window_size = window_size
        self.read_lines = 0
        self.losses = np.array([])
        self.stop = False

    def read_logs(self):
        """Read the logs and concat the array to the losses array.
        It skips the already read lines.
        """
        new_losses = np.loadtxt(self.logs_path, skiprows=self.read_lines)
        if new_losses.shape[0] > 0:
            # The loss is the last column, see trainer.py log() method of Trainer class
            new_losses = new_losses[:, -1]
            self.read_lines += new_losses.shape[0]
            self.losses = np.concatenate((self.losses, new_losses))

    def start(self, update_time=10.0):
        """Starts the plotting cycle.

        Args:
            update_time (float): Time in seconds between two updates of the graph.
        """
        self.read_logs()
        figure = plt.figure()
        axis = figure.add_subplot(1, 1, 1)
        x_data, y_data = self.get_plot_data()
        axis.set_ylim((0, np.max(y_data)))
        axis.set_xlim((0, np.max(x_data)))
        loss_line, = axis.plot(y_data)
        plt.pause(update_time)
        while not self.stop:
            self.read_logs()
            x_data, y_data = self.get_plot_data()
            loss_line.set_xdata(x_data)
            loss_line.set_ydata(y_data)
            axis.set_xlim((0, np.max(x_data)))
            figure.canvas.draw()
            plt.pause(update_time)

    def get_plot_data(self):
        """Calculates the data for plotting.
        It calculates the mean for the given window's size or return the entire losses array.

        Returns:
            np.array: x data for plotting.
            np.array: y data for plotting.
        """
        if self.mean:
            # We need to remove some items to match the reshape form
            # For example, if we have an array with 117 losses it cannot be reshaped
            # to a (1, 100) array, so we need to remove the last 17 items
            y_data = self.losses[:-(self.losses.shape[0] % self.window_size)]
            y_data = np.mean(y_data.reshape(-1, self.window_size), axis=1)
            x_data = np.arange(y_data.shape[0]) * self.window_size
            return x_data, y_data
        return np.arange(self.losses.shape[0]), self.losses

    def finish(self):
        """Set the stop flag to True to stop the updating loop."""
        self.stop = True
