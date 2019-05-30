"""Abstract trainer module.

## Quick start

A trainer has getters and methods:
- The getters are for get the different modules necessary for the training like *datasets*, *dataloaders*,
*model*, *criterion* and *optimizer*. Optionally you can provide a *logger* and a *learning rate scheduler*.
- The methods are used to optimize and evaluate the model.
  - The **train** method does the classic training algorithm.
  - The **validate** method does the validation of the model over the validation dataset.
  - The **eval** method puts the model into evaluation mode.
  - The **forward** method does a forward pass over the model and returns the loss tensor. **You must implement
    this method**.
  - The **backward** method does the backward propagation of the loss.

To use a trainer you must implement the getters methods and the *forward* method.

A good practice is to use the hyperparameters dict to store the parameters for the getters, so anyone can change
the hyperparameters without changing the code.
"""
import json
import os
import time

import torch

from torchsight.loggers import PrintLogger
from torchsight.utils import merge_dicts

LOGS_DIR = './logs'


class Trainer():
    """Base Trainer class, all the trainers must extend this class."""
    # A dict with all the hyperparameters for the different components of the training
    hyperparameters = {}

    def __init__(self, hyperparameters=None, checkpoint=None, device=None):
        """Initialize the trainer.

        Arguments:
            hyperparameters (dict, optional): A dict to change the base hyperparameters.
                If it's present, it will be deeply merged with the base hyperparameters.
        """
        base_hyperparameters = {'checkpoint': {'dir': LOGS_DIR, 'verbose': True},
                                'logger': {'dir': LOGS_DIR}}
        # Add the base hyperparameters to the trainer hyperparameters
        self.hyperparameters = merge_dicts(self.hyperparameters, base_hyperparameters)
        # Add the modified hyperparameters given in the initialization
        self.hyperparameters = merge_dicts(self.hyperparameters, hyperparameters, verbose=True)
        # Set the device of the trainer
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Get the modules for the training
        print('Loading datasets ...')
        self.dataset, self.valid_dataset = self.get_datasets()
        print('Loading dataloaders ...')
        self.dataloader, self.valid_dataloader = self.get_dataloaders()
        print('Loading model ...')
        self.model = self.get_model()
        print('Loading criterion, optimizer and scheduler ...')
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        # Load the checkpoint
        self.checkpoint = self.resume(checkpoint)
        # Get the logger
        self.logger = self.get_logger()
        # As we only log one time per batch we need to keep the state of all the elements that we want to log
        self.current_log = {}

    ####################################
    ###           GETTERS            ###
    ####################################

    def get_datasets(self):
        """Get the training and validation datasets.

        Returns:
            tuple: A Tuple with the torch.utils.data.Datasets for training and validation.
        """
        raise NotImplementedError('You must provide your own datasets.')

    def get_dataloaders(self):
        """Get the dataloaders for training and validation.

        Returns:
            tuple: The tuple with the torch.utils.data.DataLoader for training and validation.
        """
        raise NotImplementedError('You must provide the dataloaders for the datasets.')

    def get_model(self):
        """Get the model to train.

        Returns:
            torch.nn.Module: The model to train.
        """
        raise NotImplementedError('You must provide the model to train.')

    def get_criterion(self):
        """Get the criterion to use to train the model.

        Returns:
            torch.nn.Module: The criterion to use in the training.
        """
        return NotImplementedError('You must provide the criterion to train the model.')

    def get_optimizer(self):
        """Get the optimizer of the model.

        Returns:
            torch.optim.Optimizer: The optimizer of the model's parameters.
        """
        raise NotImplementedError('You must provide the optimizer to use during the training.')

    def get_scheduler(self):
        """Get the (optional) scheduler for the learning rate of the optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: A scheduler for the learning rate.
        """
        # No error because the scheduler is optional.

    def get_logger(self):
        """Get the (optional) logger to use during the training to show the information about the process.

        This base implementation uses the PrintLogger that will print the log to the console.

        Returns:
            pymatch.loggers.Logger: A Logger to use during the training.
        """
        description = 'Hyperparameters:\n{}'.format(json.dumps(self.hyperparameters, indent=2))
        return PrintLogger(description, self.hyperparameters['logger']['dir'])

    ####################################
    ###           METHODS            ###
    ####################################

    def train(self, epochs=100, validate=True):
        """Train the model during the giving epochs.

        Arguments:
            epochs (int, optional): The number of epochs to run the model.
            validate (bool, optional): If it's True the trainer will validate the training using
                the validate() method. And if there is a scheduler it gives the validation loss
                generated by the validate() method to the scheduler to adjust the learning rate.
        """
        self.model.to(self.device)

        # The criterion could be inside the model for example and in that case it could be None
        if self.criterion is not None:
            self.criterion.to(self.device)

        # The number of batches that the training dataset have
        n_batches = len(self.dataloader)

        # The start time of the training and the last batch's end time
        start_time = time.time()
        last_endtime = start_time

        # We start from the next epoch of the checkpoint (if there is any)
        start_epoch = 1 if self.checkpoint is None else self.checkpoint['epoch'] + 1

        for epoch in range(start_epoch, start_epoch + epochs):
            # Indicate to the model that we are in training mode, useful for batch normalization or dropouts modules.
            # For more info see:
            # https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
            self.model.train()

            for batch, data in enumerate(self.dataloader):
                # Optimize
                self.optimizer.zero_grad()
                loss = self.forward(*data)
                self.backward(loss)
                self.optimizer.step()

                # Log the batch
                learning_rates = [str(param_group['lr'])
                                  for i, param_group in enumerate(self.optimizer.param_groups)]

                total_time = time.time() - start_time
                batch_time = time.time() - last_endtime
                last_endtime = time.time()

                self.logger.log(merge_dicts({
                    'Training': None,
                    'Epoch': epoch,
                    'Batch': '{}/{}'.format(batch + 1, n_batches),
                    'LR': ' '.join(learning_rates),
                    'Loss': '{:.7f}'.format(float(loss)),
                    'Time': '{:.3f} s'.format(batch_time),
                    'Total': '{:.1f} s'.format(total_time)
                }, self.current_log))
                self.current_log = {}  # Restart the log dict for the next batch

                # Call the callback for the batch
                self.batch_callback(batch, epoch)

            # Call the callback for the epoch
            self.epoch_callback(epoch)

            # Save the checkpoint for this epoch
            self.save(epoch)

            if validate:
                loss = self.validate(epoch)
                if self.scheduler is not None:
                    self.scheduler.step(loss)

    def forward(self, *args):
        """Do a forward pass over the model with the model and get the loss value.

        Arguments:
            *args: All the data that the dataloader generates while iterating over it.

        Returns:
            torch.Tensor: The loss value of the forward pass.
        """
        raise NotImplementedError('You must implement the forward pass over the model.')

    def backward(self, loss):
        """Do the backward pass over the network.

        There is a method for this because each experiment could do different things during the backward like:
        ```python
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        ```

        But in this case it only does the backward of the loss.

        Arguments:
            loss (torch.Tensor): The loss value computed during the forward pass.
        """
        loss.backward()

    def eval(self):
        """Set the model into evaluation mode.

        It's a method to override this and provide a custom eval() call if you want.
        """
        self.model.eval()

    def validate(self, epoch):
        """Run the model over the validation dataset and return the mean loss over it."""
        self.model.to(self.device)
        self.eval()

        start_time = time.time()
        last_endtime = start_time

        n_batches = len(self.valid_dataloader)

        losses = []

        with torch.no_grad():
            for batch, data in enumerate(self.valid_dataloader):
                loss = float(self.forward(*data))

                batch_time = time.time() - last_endtime
                last_endtime = time.time()
                total_time = time.time() - start_time

                self.logger.log(merge_dicts({
                    'Validating': None,
                    'Epoch': epoch,
                    'Batch': '{}/{}'.format(batch + 1, n_batches),
                    'Loss': '{:.7f}'.format(float(loss)),
                    'Time': '{:.3f} s'.format(batch_time),
                    'Total': '{:.1f} s'.format(total_time)
                }, self.current_log))
                self.current_log = {}  # Restart the log dict for the next batch

                losses.append(loss)

        return torch.Tensor(losses).mean()

    def batch_callback(self, batch, epoch):
        """Method that is called after a batch has finished its process."""

    def epoch_callback(self, epoch):
        """Method that is called after an epoch has finished its process."""

    def save(self, epoch):
        """Save the checkpoint of the trainer.

        The checkpoint is a dict like:
        {'epoch': int, 'model': state_dict, 'optimizer': state_dict, 'scheduler': state_dict}
        where the scheduler is optional.

        Arguments:
            epoch (int): The epoch that has finished.
        """
        params = self.hyperparameters['checkpoint']
        path = os.path.join(params['dir'], 'checkpoint_epoch_{}.pth.tar'.format(epoch))

        if params['verbose']:
            print('[Epoch {}] Saving checkpoint to: {}'.format(epoch, path))

        checkpoint = {'epoch': epoch,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'hyperparameters': self.hyperparameters}

        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def resume(self, checkpoint):
        """Resume the training based on a last checkpoint and get the checkpoint dict.

        This method does only return the epoch value in the dict to avoid memory leaks, we don't need
        to keep the state_dicts in memory.
        You can customize your own trainer and return more values.

        Arguments:
            checkpoint (str): The path to the checkpoint file.

        Returns:
            dict: A dict with the epoch only. The state dict are not returned to not keep them
                in memory.
        """
        if checkpoint is None:
            return None

        verbose = self.hyperparameters['checkpoint']['verbose']

        if verbose:
            print('Loading checkpoint from {}'.format(checkpoint))

        checkpoint = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, val in state.items():
                if torch.is_tensor(val):
                    state[k] = val.to(self.device)

        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        return {'epoch': checkpoint['epoch']}

    @classmethod
    def from_checkpoint(cls, checkpoint, new_params=None, device=None, verbose=True):
        """Get an instance of the trainer based on the given checkpoint file.

        This is very useful because the checkpoint saves the hyperparameters too,
        so you have a trainer with the same hyperparameters that one from the checkpoint.

        Also, you can use this method to load the model, because you can do
        `trainer.model` to get the model instance.

        Arguments:
            checkpoint (str): The path to the file that contains the checkpoint file.
            new_params (dict, optional): A dict with new hyperparameters to change the ones
                in the checkpoint. Useful for example to change the batch size, the dataset root,
                etc.

        Returns:
            Trainer: An instance of the trainer with the exact same hyperparameters and with
                the modules with their state_dicts from the checkpoint too.
        """
        hyperparameters = merge_dicts(torch.load(checkpoint)['hyperparameters'], new_params, verbose)

        return cls(hyperparameters=hyperparameters, checkpoint=checkpoint, device=device)
