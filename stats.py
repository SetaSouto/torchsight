"""Script to show the loss per epoch in the validation and the training set."""
import argparse

import matplotlib.pyplot as plt
import torch

from torchsight.loggers import Logger

PARSER = argparse.ArgumentParser(
    description='Get the mean loss per epoch over the training dataset and validation dataset.'
)
PARSER.add_argument('-r', '--root', help='The root directory where are the logs files.', required=True)
PARSER.add_argument('-t', '--training-logs', help='Name of the file that contains the training logs.',
                    default='logs.json')
PARSER.add_argument('-v', '--validation-logs', help='Nome of the file that contains the validation logs.',
                    default='validation_logs.json')
PARSER.add_argument('-e', '--epoch-key', help='The key in the logs that indicates the epoch',
                    default='Epoch')
PARSER.add_argument('-l', '--loss-key', help='The key in the logs that indicates the loss of the batch.',
                    default='Total')
PARSER.add_argument('-p', '--plot', help='Plot the data using matplotlib.', action='store_const',
                    const=True, default=False)

ARGUMENTS = PARSER.parse_args()

TRAIN_LOGGER = Logger(description=None, directory=ARGUMENTS.root, filename=ARGUMENTS.training_logs)
VALID_LOGGER = Logger(description=None, directory=ARGUMENTS.root, filename=ARGUMENTS.validation_logs)

# PRINT THE DATA
TRAIN_LOSSES = TRAIN_LOGGER.epoch_losses(epoch_key=ARGUMENTS.epoch_key, loss_key=ARGUMENTS.loss_key)
EPOCHS = TRAIN_LOGGER.get_epochs(ARGUMENTS.epoch_key)

try:
    VALID_LOSSES = VALID_LOGGER.epoch_losses(epoch_key=ARGUMENTS.epoch_key, loss_key=ARGUMENTS.loss_key)
except KeyError:
    # There is no validation yet
    VALID_LOSSES = torch.zeros((len(TRAIN_LOSSES)))

PADDING = 15
print('{} | {} | {}'.format('EPOCH'.rjust(PADDING//3), 'TRAIN LOSS'.rjust(PADDING), 'VALIDATION LOSS'.rjust(PADDING)))
for index, epoch in enumerate(EPOCHS):
    try:
        train_loss = '{:.7f}'.format(float(TRAIN_LOSSES[index])).rjust(PADDING)
        valid_loss = '{:.7f}'.format(float(VALID_LOSSES[index])).rjust(PADDING)
        print('{} | {} | {}'.format('{}'.format(epoch).rjust(PADDING // 3), train_loss, valid_loss))
    except IndexError:
        continue

# PLOT THE DATA
if ARGUMENTS.plot:
    MIN_DIM = min([len(EPOCHS), len(TRAIN_LOSSES), len(VALID_LOSSES)])

    plt.subplot(1, 2, 1)
    plt.plot(EPOCHS[:MIN_DIM], TRAIN_LOSSES[:MIN_DIM].numpy())
    plt.grid(True)
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(EPOCHS[:MIN_DIM], VALID_LOSSES[:MIN_DIM].numpy())
    plt.grid(True)
    plt.title('Validation Loss')

    plt.show()
