import os
import time

import numpy as np
import torch

# TODO: Smart saving of the weights.


class Trainer:
    """A class to abstract the logic of the training. It trains the model with the given
    criterion and optimizer, create logs, validate against a validation dataset and more
    options.
    """

    def __init__(self, model, train_loader, criterion, optimizer, logs_dir=None, description=None, device=None, valid_loader=None):
        """Initialize a trainer to train and/or evaluate a model.

        The loaders must return a tuple as (input, target, *more) where more could be more arguments
        but this class only use the first element of the tuple as the input for the model and the
        second argument as the target output.

        The description of the trainer could be useful to save metadata of the training as the
        hyperparameters, conditions or anything related to the training itself or other things.
        The description is saved inside the logs directory.

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
            logs_dir (str, optional): Path to the directory where to save the logs of the training.
            description (str): Description of the training. Useful to save metadata related to the
                training as hyperparameters, conditions and other information.
            device (str): String indicating the device where to run the training.
                If cuda is available it automatically set the device to cuda.
                See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            valid_loader (torch.utils.data.DataLoader, optional): The data loader for the validation
                set.
                This is useful to keep the track of the loss over the validation set.
                The results are tracked in the log file. Or you can call the valid() method.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.valid_loader = valid_loader
        self.timestamp = time.time()
        if logs_dir:
            # Set the logs directory and create it if it does not exists
            self.logs_dir = os.path.abspath(logs_dir)
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)
            # Save the description of the trainer
            description_path = "{}/{}_description.txt".format(self.logs_dir, self.timestamp)
            with open(description_path, 'w') as file:
                file.write(description)

    def train(self, epochs=1):
        """Train the model for the given epochs.

        It calls the the epoch_callback and batch_callback methods. This is useful
        if you want to add more code to this method so you can print some stats or
        whatever you want. You must extend this trainer and implement those methods.

        Args:
            epochs (int): Number of epochs to run.
                An epoch is a full pass over all the images of the dataset.
        """
        # Move the model to the device and set to train mode, useful for batch normalization or
        # dropouts modules. For more info see:
        # https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
        self.model.to(self.device)
        self.model.train()

        print('Training started.')
        if self.logs_dir:
            print('Logs can be found at {}'.format(self.logs_dir))

        for epoch in range(epochs):
            epoch_start_time = time.time()
            last_batch_end_time = time.time()
            epoch_loss = 0
            for i, (inputs, targets, *_) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # Log the batch
                batch_time = time.time() - last_batch_end_time
                # TODO: Investigate why changing the way that it measures the time
                # changes the result. If we move the last_batch_end_time to the end
                # it does not measures the batch time well
                last_batch_end_time = time.time()
                self.log(epoch, i, batch_time, loss)
                # Increment epoch loss
                epoch_loss += loss
                # Batch callback
                self.batch_callback(epoch, i, batch_time, loss)
            # Epoch callback
            epoch_time = time.time() - epoch_start_time
            self.epoch_callback(epoch, epoch_time, epoch_loss)

    def log(self, epoch, batch_index, batch_time, loss, train=True):
        """Append to the given log's name at the logs_dir directory a new log line.

        It logs the data as:
        <timestamp> <epoch> <batch index> <batch time> <loss>
        Where:
            - date time: Date and time when this line was logged.
            - epoch: Number of the epoch that was running.
            - batch index: Index of the batch for the given epoch.
            - batch time: How many time take that batch to evaluate.
            - loss: The computed loss for the given batch.

        The log file is saved as indicated in the documentation of get_log_file_path.

        Args:
            epoch (int): The epoch that was running when the loss was calculated.
            batch_index (int): Index of the batch that was running for the given epoch.
            batch_time (float): How many time take the batch as a timestamp
                (could be getted by time.time()).
            loss (torch.Tensor): Tensor with one dimension and one item that represents the loss
                for the given epoch and batch.
            train (bool): Indicates if it is logging the train set or not. If it is false,
                the method assumes that is logging the validation set.
        """
        if self.logs_dir:
            file_path = self.get_log_file_path(
                'train' if train else 'validation')
            with open(file_path, 'a') as file:
                file.write('{} {} {} {} {}\n'.format(
                    time.time(), epoch, batch_index, batch_time, loss.item()))

    def get_log_file_path(self, dataset):
        """Returns the log's file path as <self.logs_dir>/<self.timestamp>_logs_<dataset>.txt

        Args:
            dataset (str): Indicates the dataset that is logging.
        """
        return os.path.join(self.logs_dir, '{}_logs_{}.txt'.format(self.timestamp, dataset))

    def load_logs(self, dataset):
        """Returns a tensor with shape (number of logs, 5) containing all the logs for the given
        dataset.

        Returns:
            torch.Tensor: The tensor containing all the logs.
        """
        file_path = self.get_log_file_path(dataset)
        return self.load_logs_from(file_path)

    def load_logs_from(self, path):
        """Load the logs tensor stored at path.

        Returns:
            torch.Tensor: The tensor containing all the logs.
        """
        return torch.from_numpy(np.loadtxt(path)).to(self.device)

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

    def batch_callback(self, epoch, index, batch_time, loss):
        """Callback to call every ending of a batch processing.

        Useful for add more code to the training as print statistics or whatever
        you want to execute at the end of the batch processing.

        Args:
            epoch (int): Index of the epoch that is running.
            index (int): Index of the batch that is running.
            batch_time (float): How much time (in seconds) did the batch take to be processed.
            loss (torch.Tensor): Tensor with one item as the total loss of the batch.
        """
        pass

    def epoch_callback(self, epoch, epoch_time, loss):
        """Callback to call every ending of a epoch processing.

         Useful to execute custom code after each epoch.

         Args:
            epoch (int): Index of the epoch that run.
            epoch_time (float): Total time of the epoch in seconds.
            loss (torch.Tensor): Tensor with one item that is the total loss of the epoch.
        """
        pass
