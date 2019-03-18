"""Script to show the loss per epoch in the validation and the training set."""
import argparse

from torchsight.loggers import PrintLogger

PARSER = argparse.ArgumentParser(
    description='Get the mean loss per epoch over the training dataset and validation dataset.'
)
PARSER.add_argument('-l', '--logs', help='Name of the file that contains the training logs.',
                    default='./logs.txt')
PARSER.add_argument('-ek', '--epoch-key', help='The key in the logs that indicates the epoch',
                    default='Epoch')
PARSER.add_argument('-lk', '--loss-key', help='The key in the logs that indicates the loss of the batch.',
                    default='Total')

ARGUMENTS = PARSER.parse_args()

LOGGER = PrintLogger(description=None)

# PRINT THE DATA
TRAIN_LOSSES = LOGGER.epochs_losses(filepath=ARGUMENTS.logs, epoch_key=ARGUMENTS.epoch_key,
                                    loss_key=ARGUMENTS.loss_key, keep=lambda x: x[:10] == '[Training]')
VALID_LOSSES = LOGGER.epochs_losses(filepath=ARGUMENTS.logs, epoch_key=ARGUMENTS.epoch_key,
                                    loss_key=ARGUMENTS.loss_key, keep=lambda x: x[:12] == '[Validating]')
EPOCHS = TRAIN_LOSSES.keys()

PADDING = 15
print('{} | {} | {}'.format('EPOCH'.rjust(PADDING//3), 'TRAIN LOSS'.rjust(PADDING), 'VALIDATION LOSS'.rjust(PADDING)))
for index, epoch in enumerate(EPOCHS):
    train_loss = '{:.7f}'.format(float(TRAIN_LOSSES[epoch]['average'])).rjust(PADDING)
    try:
        valid_loss = '{:.7f}'.format(float(VALID_LOSSES[epoch]['average'])).rjust(PADDING)
    except KeyError:
        valid_loss = '---'
    print('{} | {} | {}'.format('{}'.format(epoch).rjust(PADDING // 3), train_loss, valid_loss))
