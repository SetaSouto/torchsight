"""Module to fine-tune a resnet architecture on the CIFAR10 dataset.
This is an example on how it could be used the Trainer class to train a model.
"""
import os

import torch

import torchvision
from models.resnet import Resnet
from trainer import Trainer


class Cifar10Trainer(Trainer):
    """Custom trainer for a network based on resnet trained over CIFAR10 dataset."""

    def __init__(self, *args, hyperparameters=None, model_type=18, **kwargs):
        """Initialize the trainer. It needs the hyperparameters for the training.
        The hyperparameters is a dict with these keys:
            - batch_size (int)
            - epochs (int)
            - learning_rate (float)
            - momentum (float)
            - weight_decay (float)
            - shuffle (bool)
            - num_workers (int)
        It has default values for each key, see implementation.
        """
        # Set the hyperparameters for the training
        default_hyperparameters = {
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 1e-4,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "shuffle": True,
            "num_workers": 2
        }
        if hyperparameters:
            self.hyperparameters = {**default_hyperparameters, **hyperparameters}
        else:
            self.hyperparameters = default_hyperparameters
        # Set the logs dir
        data_root = './data'
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        logs_dir = './logs/cifar10'
        # Set the parameters for this dataset
        num_classes = 10
        image_size = 32
        # Transformation for the dataset
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Datasets and data loaders
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                                transform=transformations)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=self.hyperparameters["batch_size"],
                                                   shuffle=self.hyperparameters["shuffle"],
                                                   num_workers=self.hyperparameters["num_workers"])
        validset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                transform=transformations)
        valid_loader = torch.utils.data.DataLoader(validset,
                                                   batch_size=self.hyperparameters["batch_size"],
                                                   shuffle=False,
                                                   num_workers=self.hyperparameters["num_workers"])
        # Get the model, criterion and optimizer
        model = Resnet(type=model_type, num_classes=num_classes,
                       image_size=image_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=self.hyperparameters["learning_rate"],
                                    momentum=self.hyperparameters["momentum"],
                                    weight_decay=self.hyperparameters["weight_decay"])
        # Initialize the trainer
        description = "\n".join(["{}: {}".format(key, value)
                                 for key, value in self.hyperparameters.items()])
        super().__init__(model=model,
                         train_loader=train_loader,
                         criterion=criterion,
                         optimizer=optimizer,
                         logs_dir=logs_dir,
                         description=description,
                         valid_loader=valid_loader,
                         *args, **kwargs)
        # Parameters for the callbacks
        self.batches_total_loss = 0
        self.batches_total_time = 0
        self.print_every_batches = 100

    def train(self):
        """Train the model using the epochs given in the hyperparameters."""
        super().train(self.hyperparameters["epochs"])

    def batch_callback(self, epoch, index, batch_time, loss):
        """Prints some accumulated statistics every some batches."""
        self.batches_total_loss += loss
        self.batches_total_time += batch_time
        if index % self.print_every_batches == self.print_every_batches - 1:
            mean_batch_loss = self.batches_total_loss / self.print_every_batches
            print("[{}\t{}]\tTotal time: {}\tMean loss: {}".format(
                epoch + 1, index + 1, self.batches_total_time, mean_batch_loss))
            self.batches_total_loss = 0
            self.batches_total_time = 0

    def epoch_callback(self, epoch, epoch_time, loss):
        """Prints the precision over the validation dataset."""
        # Re initialize the stats for the next batches
        self.batches_total_loss = 0
        self.batches_total_time = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('[Epoch: {}\tTime: {}\tLoss: {}] Accuracy of the network on the 10000 test images: {}%'.format(
            epoch + 1, epoch_time, loss, 100 * correct / total))

if __name__ == "__main__":
    Cifar10Trainer().train()
