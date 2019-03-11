"""Base trainer module."""
import json
import os
import time

import torch

from ..loggers import Logger


class AbstractTrainer():
    """Abstract trainer to train a pytorch model.

    To create a trainer please override the getters methods and the train method.

    Attributes:
        hyperparameters (dict): The hyperparameters for the training.
            Here you can add the 'learning_rate' hyperparameter.
        dataset (torch.utils.data.Dataset): The dataset for training.
        valid_dataset (torch.utils.data.Dataset): The validation dataset.
        device (str): The device where is running the training.
        model (torch.nn.Module): The model that is training.
        dataloader (torch.utils.data.Dataloader): The dataloader for the training dataset.
        valid_dataloader (torch.utils.data.Dataloader): The dataloader for the validation dataset.
        criterion (torch.nn.Module): The criterion -or loss- function for the training.
        optimizer (torch.optim.Optimizer): The optimizer for the training.
    """

    hyperparameters = {}  # Base hyperparameters

    def __init__(self, hyperparameters, logs_dir='./logs', checkpoint=None):
        """Initialize the trainer. Sets the hyperparameters for the training.

        Args:
            hyperparameters (dict): The hyperparameters for the training.
            logs_dir (str): Path to the directory where to save the logs.
            checkpoint (str): Path to a checkpoint dict.
        """
        print('\n--------- TRAINER ----------\n')
        self.hyperparameters = self.merge_hyperparameters(self.hyperparameters, hyperparameters)

        # Set the datasets
        self.dataset, self.valid_dataset = self.get_datasets()

        # Set the model, data loaders, criterion and optimizer for the training
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.get_model()
        self.dataloader, self.valid_dataloader = self.get_dataloaders()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()

        # Load the checkpoint if there is any
        if checkpoint is not None:
            self.checkpoint_epoch = self.resume(checkpoint)
        else:
            self.checkpoint_epoch = 0

        # Configure the logs
        self.logger = None
        if logs_dir:
            logs_dir = os.path.join(logs_dir, str(int(time.time())))

            # Description of this instance are the hyperparameters of the training
            description = ['Hyperparameters:\n', json.dumps(self.hyperparameters, indent=2)]
            if checkpoint is not None:
                description.append('\nCheckpoint: {}'.format(checkpoint))

            self.logger = Logger(description='\n'.join(description), directory=logs_dir)

    def merge_hyperparameters(self, base, new, path=None):
        """Merge the base hyperparameters (if there's any) with the given hyperparameters.

        Arguments:
            hyperparameters (dict): The modified hyperparameters.

        Returns:
            dict: The deeply merged hyperparameters.
        """
        if path is None:
            path = []

        for key in new:
            if key in base:
                if isinstance(base[key], dict) and isinstance(new[key], dict):
                    self.merge_hyperparameters(base[key], new[key], path + [str(key)])
                elif base[key] == new[key]:
                    pass  # same leaf value
                else:
                    print('INFO: Replacing base hyperparameter "{}" with value < {} > for < {} >.'.format(key,
                                                                                                          base[key],
                                                                                                          new[key]))
                    base[key] = new[key]
            else:
                print('Warn: Hyperparameter "{key}" not present in base hyperparameters.'.format(key=key))
        return base

    def train(self):
        """Train the model. You must implement this method according to your model."""
        raise NotImplementedError()

    def save_checkpoint(self, epoch):
        """Save the training's parameters.

        It saves the model and the optimizer.

        Arguments:
            epoch (int): The epoch when this checkpoint was generated.
        """
        path = os.path.join(self.logger.directory, 'checkpoint_epoch_{}.pth.tar'.format(epoch))
        print("[Epoch {}] Saving model's and optimizer's state dict to: {}".format(epoch, path))
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def resume(self, checkpoint):
        """Load the state dicts of the model and optimizer.

        Arguments:
            checkpoint (str): The absolute path to the checkpoint file.

        Returns:
            int: The epoch when the checkpoint was saved.
        """
        print('Loading checkpoint from {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, val in state.items():
                if torch.is_tensor(val):
                    state[k] = val.to(self.device)
        return checkpoint['epoch']

    def validate(self):
        """Validation method.

        It's called after each epoch. Use it like you want.
        """
        pass

    # -----------------------------------
    #              GETTERS
    # -----------------------------------

    def get_model(self):
        """Initialize and return the model to train.

        Returns:
            (torch.nn.Module): The encoder that will generate the embeddings.
        """
        raise NotImplementedError()

    def get_criterion(self):
        """Returns the criterion (or loss) for the training.

        Returns:
            (nn.Module): The loss for the training.
        """
        raise NotImplementedError()

    def get_transform(self):
        """Get the transform to apply over the dataset.

        Returns:
            (torchvision.transforms.Compose): The transform to apply over the dataset.
        """
        raise NotImplementedError()

    def get_datasets(self):
        """Get the training and validation datasets.

        Returns:
            (torch.utils.data.Dataset): The dataset for training.
            (torch.utils.data.Dataset): The dataset for validation.
        """
        raise NotImplementedError()

    def get_dataloaders(self):
        """Returns the train and validation dataloaders.

        Returns:
            (torch.utils.data.DataLoader): The dataloader that will sample the elements from the training dataset.
            (torch.utils.data.DataLoader): The dataloader that will sample the elements from the validation dataset.
        """
        raise NotImplementedError()

    def get_optimizer(self):
        """Get the optimizer of the training.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer of the training.
                For the optimizer package see: https://pytorch.org/docs/stable/optim.html
        """
        raise NotImplementedError()
