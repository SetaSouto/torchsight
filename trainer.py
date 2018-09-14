import os
import time

import numpy as np
import torch


class Trainer:
    """A class to abstract the logic of the training. It trains the model with the given
    criterion and optimizer, create logs, validate against a validation dataset and more
    options.
    """

    def __init__(self, model, train_loader, criterion, optimizer, device=None, valid_loader=None,
                 logs_dir=None):
        """Initialize a trainer to train and/or evaluate a model.

        The loaders must return a tuple as (input, target, *more) where more could be more arguments
        but this class only use the first element of the tuple as the input for the model and the
        second argument as the target output.

        Args:
            model (torch.nn.Module): The model created with pytorch to train.
            train_loader (torch.utils.data.DataLoader): The data loader of the train set.
            criterion (torch.nn.Module): Criterion to train the model.
                This function calculate the loss of the model while training.
                To use a custom criterion you must implement a nn.Module with its forward method.
                See:
                https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/3
                For all the loss functions already implemented in pytorch see:
                https://pytorch.org/docs/stable/nn.html
            optimizer (torch.optim.Optimizer): The optimizer of the training.
                (E.g. Stochastic Gradient Descent).
                For the optimizer package see: https://pytorch.org/docs/stable/optim.html
            device (str): String indicating the device where to run the training.
                If cuda is available it automatically set the device to cuda.
                See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            valid_loader (torch.utils.data.DataLoader, optional): The data loader for the validation
                set.
                This is useful to keep the track of the loss over the validation set.
                The results are tracked in the log file. Or you can call the valid() method.
            logs_dir (str, optional): Path to the directory where to save the logs of the training.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.valid_loader = valid_loader
        self.start_time = None  # start time of the training
        if logs_dir:
            self.logs_dir = os.path.abspath(logs_dir)
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)

    def train(self, epochs=1, epoch_callback=None, batch_callback=None):
        """Train the model for the given epochs.

        Args:
            epochs (int): Number of epochs to run.
                An epoch is a full pass over all the images of the dataset.
            epoch_callback (function): Function call every time that an epoch
                finishes.
            batch_callback (function): A function that is called with the arguments
                of the log method (without the name, that is obviously 'train')
                after each batch process. See log() for documentation.
        """
        # Move the model to the device and set to train mode, useful for batch normalization or
        # dropouts modules.
        # For more info see:
        # https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
        self.model.to(self.device)
        self.model.train()

        self.start_time = time.time()

        print('Training started.')
        if self.logs_dir:
            print('Logs can be found at {}'.format(self.logs_dir))

        for epoch in range(epochs):
            batch_start_time = time.time()
            for i, (inputs, targets, *_) in enumerate(self.train_loader):
                # Get input, targets and train
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # Log the batch
                batch_time = time.time() - batch_start_time
                # We need to put here the the batch start time to get the time
                # of loading the data
                batch_start_time = time.time()
                self.log(epoch, i, batch_time, loss, 'train')
                # Batch callbacks
                if batch_callback:
                    batch_callback(epoch, i, batch_time, loss)
            # Epoch callback
            if epoch_callback:
                epoch_callback()
            # Validate one time per epoch
            self.validate(epoch)

    def log(self, epoch, batch_index, batch_time, loss, name):
        """Append to the given log's name at the logs_dir directory a new log line.

        It logs the data as:
        <timestamp> <epoch> <batch index> <batch time> <loss>
        Where:
            - date time: Date and time when this line was logged.
            - epoch: Number of the epoch that was running.
            - batch index: Index of the batch for the given epoch.
            - batch time: How many time take that batch to evaluate.
            - loss: The computed loss for the given batch.

        The log file is saved as indicated in the documentation of _get_log_file_path.

        Args:
            epoch (int): The epoch that was running when the loss was calculated.
            batch_index (int): Index of the batch that was running for the given epoch.
            batch_time (float): How many time take the batch as a timestamp
                (could be getted by time.time()).
            loss (torch.Tensor): Tensor with one dimension and one item that represents the loss
                for the given epoch and batch.
            name (str): Name of the log. Useful to set the file's name.
        """
        if self.logs_dir:
            file_path = self._get_log_file_path(name)
            with open(file_path, 'a') as file:
                file.write('{} {} {} {} {}\n'.format(
                    time.time(), epoch, batch_index, batch_time, loss.item()))

    def _get_log_file_path(self, name):
        """Returns the log's file path as <self.logs_dir>/logs_<name>_<self.start_time>.txt"""
        return os.path.join(self.logs_dir, 'logs_{}_{}.txt'.format(name, self.start_time))

    def load_logs(self, name):
        """Returns a tensor with shape (number of logs, 5) containing all the logs for the given
        log name.

        Returns:
            torch.Tensor: The tensor containing all the logs.
        """
        file_path = self._get_log_file_path(name)
        return self.load_logs_from(file_path)

    @staticmethod
    def load_logs_from(path):
        """Load the logs tensor stored at path.

        Returns:
            torch.Tensor: The tensor containing all the logs.
        """
        return torch.from_numpy(np.loadtxt(path))

    def validate(self, epoch):
        """Validate the model to the given valid dataset.
        If there is no valid dataset it does not do any validation.
        It logs the validation with name 'validation'.
        """
        if self.valid_loader:
            total_loss = 0.0
            start_time = time.time()
            for i, (inputs, targets, *_) in enumerate(self.valid_loader):
                # Get input, targets and train
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets).item()
            # Log the validation
            self.log(epoch, 0, time.time() - start_time,
                     total_loss, 'validation')
