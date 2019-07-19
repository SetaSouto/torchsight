"""A logger that only uses the print() function and has static methods to read those lines."""
import os

from .abstract import AbstractLogger


class PrintLogger(AbstractLogger):
    """Log the data only using the print() function.

    How can we store the values? Calling the script and setting the output to a file.
    Example:
    python train.py > logs.txt

    So this Logger class can be used to parse those logs and get information using static methods.

    A good practice would be having an already created directory where to store the logs and initialize
    this logger with that directory and output the stdout to a file inside that directory.
    """

    def __init__(self, description=None, directory=None):
        """Initialize the logger.

        Arguments:
            description (str, optional): A description to save in the directory as a txt file.
                Useful to store the hyperparameters of the training for example.
            directory (str, optional): The directory where to save the description file.
        """
        if directory is not None and not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory

        if description is not None and directory is not None:
            with open(os.path.join(directory, 'description.txt'), 'w') as file:
                file.write(description)

    def log(self, data):
        """Log the data dict.

        It generates a line using print() that has every key-value pair of the dict like:
        [key1 value1] [key2 value2] ... [key_n value_n]

        If you want to print only a key (like 'validating') you can pass a None value.

        Arguments:
            data (dict): A dictionary with key-value pairs to log.
        """
        items = []
        for key, value in data.items():
            if value is None:
                items.append('[{}]'.format(key))
            else:
                items.append('[{} {}]'.format(key, value))
        print(' '.join(items))

    @staticmethod
    def read(filepath, keep=None):
        """Read a file that could contain lines generated by this logger.

        The file could have lines from other modules by this logger will only take the lines that start with '[' and
        finishes with ']'.

        Arguments:
            filepath (str): The path to the file that contains the lines generated by this logger.
            keep (function, optional): A function that returns True if the line must be keeped
                or False if not.
                Example: A line could be '[Validating] [Epoch 10] ...' so you can implement a lambda like
                lambda x: x[:13] == '[Validating]' to return only the lines of the validation.

        Returns:
            list: A list with each logged dict.
        """
        with open(filepath, 'r') as file:
            lines = file.read().split('\n')

        # Clean the not logs lines
        logs = []
        for line in lines:
            if not line:
                continue

            if line[0] != '[' or line[-1] != ']':
                continue

            if keep is not None and not keep(line):
                continue

            line = line[1:-1]  # Remove the first '[' and the last ']'
            pairs = line.split('] [')  # Get the key-value pairs
            current = {}  # Current log dict
            for pair in pairs:
                try:
                    key, value = pair.split(' ')
                    current[key] = value
                except ValueError:
                    # There is only a key without a value
                    current[pair] = None
            logs.append(current)

        return logs

    @staticmethod
    def epochs_losses(filepath, epoch_key='epoch', loss_key='loss', keep=None):
        """Get the average loss per epoch given a logs files.

        Arguments:
            filepath (str): The path to the file that contains the lines generated by this logger.
            epoch_key (str, optional): The key of the epoch in the log dict.
            loss_key (str, optional): The key of the loss in the log dict.
            keep (function, optional): See read() method.

        Returns:
            dict: Dict with epoch as key and an other dict as value with 'sum', 'count' and 'average'.
                Where 'sum' is the sum of the losses of the epochs, 'count' is how many logs does the
                epoch have and 'average' is simply 'sum' divided by 'count'.
        """
        losses = {}  # Initialize the return value

        for log in PrintLogger.read(filepath, keep):
            epoch = log[epoch_key]
            try:
                loss = log[loss_key]
            except KeyError:
                continue
            if epoch not in losses:
                losses[epoch] = {'sum': 0, 'count': 0}
            losses[epoch]['sum'] += float(loss)
            losses[epoch]['count'] += 1
        # Get the average for each epoch
        for epoch in losses:
            losses[epoch]['average'] = losses[epoch]['sum'] / losses[epoch]['count']

        return losses
