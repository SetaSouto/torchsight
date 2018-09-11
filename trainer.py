import numpy as np
import os
import time
import torch


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device=None, valid_loader=None, logs_dir=None):
        """Initialize a trainer to train and/or evaluate a model.

        Args:
            model (torch.nn.Module): The model created with pytorch to train.
            train_loader (torch.utils.data.DataLoader): The data loader of the train set.
            criterion (torch.nn.Module): Criterion to train the model.
                This function calculate the loss of the model while training.
                To use a custom criterion you must implement a nn.Module with its forward method. See:
                https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/3
                For all the loss functions already implemented in pytorch see:
                https://pytorch.org/docs/stable/nn.html
            optimizer (torch.optim.Optimizer): The optimizer of the training. (E.g. Stochastic Gradient Descent).
                For the optimizer package see: https://pytorch.org/docs/stable/optim.html
            device (str): String indicating the device where to run the training.
                If cuda is available it automatically set the device to cuda.
                See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            valid_loader (torch.utils.data.DataLoader, optional): The data loader for the validation set.
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
        self.start_time = None # start time of the training
        if logs_dir:
            self.logs_dir = os.path.abspath(logs_dir)
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)

    def train(self, epochs=1):
        """Train the model for the given epochs.

        Args:
            epochs (int): Number of epochs to run.
                An epoch is a full pass over all the images of the dataset.
        """
        # Move the model to the device and set to train mode, useful for batch normalization or dropouts modules.
        # For more info see: https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
        self.model.to(self.device)
        self.model.train()

        self.start_time = time.time()

        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                self.log(epoch, i, loss)

    def log(self, epoch, batch_index, batch_time, loss, name="train"):
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
            batch_time (float): How many time take the batch as a timestamp (could be getted by time.time()).
            loss (torch.Tensor): Tensor with one dimension and one item that represents the loss
                for the given epoch and batch.
            name (str): Name of the log. Useful to set the file's name.
        """
        if self.logs_dir:
            file_path = self._get_log_file_path(name)
            with open(file_path, 'a') as file:
                file.write('{} {} {} {} {}\n'.format(time.time(), epoch, batch_index, batch_time, loss))
    
    def _get_log_file_path(self, name):
        """Returns the log's file path as <self.logs_dir>/logs_<name>_<self.start_time>.txt"""
        return os.path.join(self.logs_dir, 'logs_{}_{}.txt'.format(name, self.start_time))

    def load_logs(self, name):
        """Returns a tensor with shape (number of logs, 5) containing all the logs for the given log name.

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

    def validate(self):
        pass
