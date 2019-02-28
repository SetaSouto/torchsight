"""Base Logger module."""
import json
import os

import torch


class Logger():
    """Base Logger class."""

    def __init__(self, description=None, directory='./logs', filename='logs.json'):
        """Initialize the logger and create the directory that will contain the logs.

        Arguments:
            description (str): Description of the model / training or whatever you want to
                save as free text. It saves this string inside the directory with the filename
                'description.txt'.
            dir (str): Path to the directory that will contain the different log files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory
        self.log_file = os.path.join(self.directory, filename)

        if description is not None:
            with open(os.path.join(self.directory, 'description.txt'), 'w') as file:
                file.write(description)

    def log(self, data):
        """Log the data.

        Arguments:
            data (dict): A dict with the data to log.
        """
        self._print(data)
        self._append(data)

    def _print(self, data):
        """Print the data dict in the console.

        It renders a line as [<key> <value>] [ ... ].

        Example:
        A dict like data = {'Batch': 5, 'loss': 0.874} will render:
        [Batch 5] [loss 0.874]

        Arguments:
            data (dict): The data to log.
        """
        log = ['[{} {}]'.format(key, value) for key, value in data.items()]
        print(' '.join(log))

    def _append(self, data):
        """Append the values of the data dict to the log file.

        Example:
        For a data dict like {'Batch': 1, 'loss': 0.4353} it will generate a file like:
        {
            'Batch': [1],
            'loss': [0.4353]
        }
        And all the next logs will append the values to its correspondent key, resulting in
        something like:
        {
            'Batch': [1, 2, 3],
            'loss': [0.4352, 0.34223, 0.24323]
        }

        Arguments:
            data (dict): The dict with the data to append to the log file.
        """
        logs = self.read()

        for key, value in data.items():
            if not key in logs:
                logs[key] = []
            logs[key].append(value)

        with open(self.log_file, 'w') as file:
            file.write(json.dumps(logs, indent=2))

    def read(self):
        """Read the logs from the file (if exists) and return the dict.
        If there is no file it returns an empty dict.

        Returns:
            dict: The json saved with the logs.
        """
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as file:
                return json.loads(file.read())
        return {}

    def average_loss(self, key='loss', window=1e3):
        """Average the loss with the given window size and show it.

        Basically it gets the loss from the log file with the given key, average the values for each
        window consecutive times and returns the reduced array.
        For example, if we have 1e5 values for the loss and a window size of 1e3 it average each 1e3
        losses and return an array with 1e2 values.

        If the loss array does not have a multiply of the window size it cuts the oldest values.

        Return:
            torch.Tensor: The tensor with the averaged loss with the given window size.
        """
        losses = torch.Tensor([float(val) for val in self.read()[key]])

        n_losses = losses.shape[0]
        if n_losses % window != 0:
            losses = losses[int(n_losses % window):]

        losses = losses.view(-1, window)
        return losses.mean(dim=1)
