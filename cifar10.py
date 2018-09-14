"""Module to fine-tune a resnet architecture on the CIFAR10 dataset"""
import os

import torch

import torchvision
from models.resnet import Resnet
from trainer import Trainer


def main():
    """Main function to train a resnet module on the CIFAR10 dataset."""
    data_root = './data'
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    logs_dir = './logs/cifar10'
    num_classes = 10
    image_size = 32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type = 18  # Resnet18
    print_every = 100  # Print info every 100 batches
    # Hyperparameters
    batch_size = 64
    epochs = 10
    learning_rate = 1e-3
    momentum = 0.9
    shuffle = True
    num_workers = 2

    # Transformation for the dataset
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Datasets and data loaders
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                            transform=transformations)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    validset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                            transform=transformations)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
    # Get the model, criterion and optimizer
    model = Resnet(type=model_type, num_classes=num_classes,
                   image_size=image_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)
    # Initialize the trainer
    trainer = Trainer(model=model, train_loader=train_loader, criterion=criterion,
                      optimizer=optimizer, device=device, logs_dir=logs_dir)

    def validset_precision():
        """Calculates the precision over the validset."""
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {}%'.format(
            100 * correct / total))

    class BatchCallback:
        """A class to keep the state of the batch callback."""

        def __init__(self):
            """Initialize the total loss."""
            self.total_loss = 0
            self.total_time = 0

        def get_callback(self):
            """Function to call every time after each batch process.
            It prints some statistics."""
            def function(epoch, batch_index, batch_time, loss):
                self.total_loss += loss
                self.total_time += batch_time
                if batch_index % print_every == print_every - 1:
                    mean_batch_loss = self.total_loss / print_every
                    print("[{}\t{}]\tTotal time: {}\tMean loss: {}".format(
                        epoch + 1, batch_index + 1, self.total_time, mean_batch_loss))
                    self.total_loss = 0
                    self.total_time = 0
            return function

    # TRAINING
    trainer.train(epochs=epochs, epoch_callback=validset_precision,
                  batch_callback=BatchCallback().get_callback())


if __name__ == "__main__":
    main()
