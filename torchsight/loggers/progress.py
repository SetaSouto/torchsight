"""Module with the implementation of the Logger that uses tqdm to show a progress bar for the training."""
import os

from tqdm import tqdm


class ProgressMetric():
    """An instance of a metric in an epoch, it holds the settings of the metrics for the ProgressLogger."""

    def __init__(self, name, accumulate=True, reduce='avg', template='{:.5f}'):
        """Initialize the metric.

        Arguments:
            name (str): the name of the metric.
            accumulate (bool, optional): indicates that the value when updated must be accumulated.
            reduce (str, optional): the reduce method to use. It could be 'avg' or 'sum'. Could not be used with
                `accumulate=False`.
        """
        self.name = name
        self.accumulate = accumulate
        self.reduce = reduce
        self.template = template
        self.value = None
        self.update_value = self.value  # Keep a copy of the last value that updated the metric
        self.reset()

    def reset(self):
        """Reset the value of the metric."""
        self.value = 0 if self.accumulate else None
        self._update_value = self.value

    def update(self, value):
        """Update the value of the metric.

        Arguments:
            value: the value to accumulate or replace.
        """
        self.update_value = value
        if self.accumulate:
            self.value += value
        else:
            self.value = value

    def get(self, batches=None):
        """Get the metric value.

        If the value is accumulated and the reduce method is 'avg' you must provide the batches
        argument to divide the accumulated value between the batches to take the average.

        Arguments:
            batches (int, optional): the number of batches to take the average of the metric.
        """
        if self.accumulate and self.reduce == 'avg':
            if batches is None:
                raise ValueError('Using "avg" reduce, provide the batches to take the average of the metric.')
            return self.template.format(self.value / batches)

        return self.template.format(self.value)


class ProgressLogger():
    """A logger for any process or experiment that needs to store and log metrics per batches and epochs.

    An important note is that the logger counts the batches and epochs starting from 1, not 0.

    Usage:

    ```python
    import time
    from torchsight.loggers.progress import ProgressLogger, ProgressMetric

    logger = ProgressLogger(metrics=[
        ProgressMetric(name='Loss'),
        ProgressMetric(name='Positive'),
        ProgressMetric(name='Learning rate', accumulate=False)
    ])
    for epoch in logger.epochs(100):
        for batch, data in logger.batches(range(100)):
            time.sleep(1)
            logger.set_metrics({
                'Loss': 100 - batch,
                'Positive': 50 - batch/2,
                'Learning rate': 0.1,
            })
    ```
    """

    def __init__(self, metrics, output_dir='.', batches_file='batches_metrics.csv', epochs_file='epochs_metrics.csv',
                 append=False, overwrite=False):
        """Initialize the logger.

        Arguments:os.path.join(output_dir, fos.path.join(output_dir, file)ile)
            metrics (list of ProgressMetrics): with the metrics to keep track.
            output_dir (str, optional): the output directory of the logger.
            batches_file (str, optional): the name of the CSV file that will be in the output directory
                with the metrics per batch.
            epochs_file (str, optional): the name of the CSV file that will be in the output directory
                with the metrics per batch.
            append (bool, optional): append the new metrics to the files if they already exists.
            overwrite (bool, optional): overwrite the files with this new ones.
        """
        if any(m for m in metrics if not isinstance(m, ProgressMetric)):
            raise ValueError('Please provide only ProgressMetric instances in the metrics argument.')

        self.metrics = {m.name: m for m in metrics}  # Keyed by name
        self.output_dir = output_dir

        # Check if the files already exists to avoid problems
        for file in [batches_file, epochs_file]:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path) and not append and not overwrite:
                raise ValueError(f'File "{file_path}" already exists, you probably want to backup its data. '
                                 'Is that is not the case please provide the "append" or "overwrite" flag as True.')

        self.batches_file = os.path.join(output_dir, batches_file)
        self.epochs_file = os.path.join(output_dir, epochs_file)
        self.epoch = None
        self.batch = None
        self.epochs_bar = None
        self.batches_bar = None

        # Add the headers for the CSV files
        if overwrite or not append:
            with open(self.batches_file, 'w') as file:
                file.write(','.join(['Epoch', 'Batch', *self.metrics.keys()]) + '\n')
            with open(self.epochs_file, 'w') as file:
                file.write(','.join(['Epoch', *self.metrics.keys()]) + '\n')

    def epochs(self, num_epochs, start_epoch=1):
        """Return a generator that renders a progress bar for the whole training and yields the epoch
        number.

        Arguments:
            num_epochs (int): the number of epochs to run.
            start_epoch (int, optional): the starting epoch number. Useful when resuming an old training
                for example.

        Returns:
            Generator: with that yields the number of the epoch.
        """
        self.epochs_bar = tqdm(range(start_epoch, start_epoch + num_epochs))

        def generator():
            for epoch in self.epochs_bar:
                self.epoch = epoch
                if epoch > start_epoch:
                    # Save the metrics of the the last epoch
                    self.save_epoch_metrics()
                    # Restart the metrics
                    for m in self.metrics.values():
                        m.reset()
                # Yield the number of the epoch
                yield epoch

        return generator()

    def batches(self, data_loader):
        """Return a generator that renders a progress bar for the progress of the current epoch.

        Arguments:
            data_loader (torch.utils.data.DataLoader): to load the data and generate the batches.

        Returns:
            Generator: that yields the batch number (1-indexed) and batch's data generated with the
                DataLoader.
        """
        self.batches_bar = tqdm(data_loader)

        def generator():
            for i, data in enumerate(self.batches_bar):
                self.batch = i + 1  # Set the current batch
                if self.batch > 1:
                    # Save the metrics of the batch
                    self.save_batch_metrics()
                # Yield the number of the batch and its data
                yield self.batch, data

        return generator()

    def set_metrics(self, keyed_values):
        """Set the metrics' values keyed by their names.

        Arguments:
            keyed_values (dict): with the value for each metric keyed by the metrics name.
        """
        for name, value in keyed_values.items():
            if name not in self.metrics:
                raise ValueError(f'The metric with name "{name}" was not declared in this logger.')
            self.metrics[name].update(value)

        self.update_epochs_description()
        self.update_batches_description()

    @property
    def epochs_description(self):
        """A property to get the description of the epochs bar."""
        description = f'[Epoch {self.epoch}] '
        description += ' '.join([f'[{m.name} {m.get(self.batch)}]' for m in self.metrics.values()])
        return description

    def update_epochs_description(self):
        """Update the description of the epochs bar with the reduced metrics."""
        self.epochs_bar.set_description(self.epochs_description)

    def update_batches_description(self):
        """Update the description of the batches bar with the last updated values of the metrics."""
        if self.batch == len(self.batches_bar):
            self.batches_bar.set_description(self.epochs_description)
        else:
            description = f'[Batch {self.batch}] '
            description += ' '.join([f'[{m.name} {m.update_value}]' for m in self.metrics.values()])
            self.batches_bar.set_description(description)

    @staticmethod
    def append_metrics(file, values):
        """Append the given values to the given file.

        Arguments:
            file(str): the path of the file to append the values.
            values(list): with the values to append.
        """
        with open(file, 'a') as f:
            f.write(','.join([str(v) for v in values]) + '\n')

    def save_batch_metrics(self):
        """Save the metrics of the current batch."""
        self.append_metrics(self.batches_file, [self.epoch, self.batch,
                                                *[m.update_value for m in self.metrics.values()]])

    def save_epoch_metrics(self):
        """Save the reduced metrics of the current epoch."""
        batches = len(self.batches_bar)
        self.append_metrics(self.epochs_file, [self.epoch, *[m.get(batches) for m in self.metrics.values()]])
