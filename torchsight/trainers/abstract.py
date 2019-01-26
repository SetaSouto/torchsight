"""Base trainer module."""
import os
import time

import torch


class AbstractTrainer():
    """Abstract trainer to train a pytorch object detection model.

    To create a trainer please override the getters methods.

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

    def __init__(self, hyperparameters, logs_dir='./logs'):
        """Initialize the trainer. Sets the hyperparameters for the training.

        Args:
            hyperparameters (dict): The hyperparameters for the training.
            logs_dir (string): Path to the directory where to save the logs.
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

        # Configure the logs
        self.logs_dir = os.path.join(logs_dir, str(int(time.time()))) if logs_dir else None
        if self.logs_dir:
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)
            # Description of this instance are the hyperparameters of the training
            description = "\n".join(["{}: {}".format(key, value) for key, value in self.hyperparameters.items()])
            with open(os.path.join(self.logs_dir, 'description.txt'), 'w') as file:
                file.write(description)

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

    def train(self, epochs=100):
        """Train the model for the given epochs.

        Args:
            epochs (int): Number of epochs to run. An epoch is a full pass over all the images of the dataset.
        """
        self.model.to(self.device)

        print('----- Training started ------')
        print('Using device: {}'.format(self.device))

        if self.logs_dir:
            print('Logs can be found at {}'.format(self.logs_dir))

        for epoch in range(epochs):

            last_endtime = time.time()

            epoch_losses = {'regression': 0, 'classification': 0, 'total': 0}

            # Set model to train mode, useful for batch normalization or dropouts modules. For more info see:
            # https://discuss.pytorch.org/t/trying-to-understand-the-meaning-of-model-train-and-model-eval/20158
            self.model.train()

            for batch_index, (images, annotations, *_) in enumerate(self.dataloader):
                images, annotations = images.to(self.device), annotations.to(self.device)

                # Optimize
                self.optimizer.zero_grad()
                anchors, regressions, classifications = self.model(images)
                anchors = anchors.to(self.device)
                del images
                classification_loss, regression_loss = self.criterion(anchors, regressions, classifications,
                                                                      annotations)
                del anchors, regressions, classifications, annotations

                loss = classification_loss + regression_loss
                # Set as float to free memory
                classification_loss = float(classification_loss)
                regression_loss = float(regression_loss)
                # Optimize
                loss.backward()
                self.optimizer.step()

                # Increment epoch loss
                epoch_losses['classification'] += classification_loss
                epoch_losses['regression'] += regression_loss
                epoch_losses['total'] += float(loss)

                # Log the batch
                batch_time = time.time() - last_endtime
                last_endtime = time.time()
                string = '[Epoch: {}] [Batch: {}] [Time: {:.3f}] '.format(epoch, batch_index, batch_time)
                string += '[Classification: {:.5f}] [Regression: {:.5f}] [Total: {:.5f}]'.format(classification_loss,
                                                                                                 regression_loss,
                                                                                                 loss)
                print(string)

            # Save weights
            self.save_weights(epoch)

            # Validate
            self.validate()

    def save_weights(self, epoch):
        """Save the model state dict."""
        path = os.path.join(self.logs_dir, 'epoch_{}_model.pth'.format(epoch))
        print("[Epoch {}] Saving model's state dict to: {}".format(epoch, path))
        self.model.save_state(path)

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
