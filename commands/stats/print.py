"""Extract stats from the PrintLogger."""
import click

from torchsight.loggers import PrintLogger


@click.command()
@click.argument('log_file')
@click.option('-lk', '--loss-key', default='Loss', help='The key in the logs that indicates the loss of the batch.')
@click.option('-ek', '--epoch-key', default='Epoch', help='The key in the logs that indicates the epoch.')
def printlogger(log_file, loss_key, epoch_key):
    """Get the mean loss per epoch over the training dataset and validation dataset.

    This stats could be generated from the logs that generated the PrintLogger and were
    saved in a file.

    For example:
    > python cli.py train dldenet ~/dataset/coco --logs-dir logs/ > logs/logs.txt

    This command indicates that the trainer save the checkpoints to logs/ and the logger
    save the description into logs/description.txt; and with the `>` we put all the prints
    into the logs/logs.txt file.

    Now you can use:
    > python cli.py stats printlogger logs/logs.txt

    And you'll get something like:
    ```
    EPOCH |      TRAIN LOSS | VALIDATION LOSS
        1 |       1.9204515 |       1.1482288
        2 |       1.1309012 | ---
    ```

    Where the `---` means that there is no data yet to show.
    """
    logger = PrintLogger(description=None)
    train = logger.epochs_losses(filepath=log_file, epoch_key=epoch_key,
                                 loss_key=loss_key, keep=lambda x: x[:10] == '[Training]')
    valid = logger.epochs_losses(filepath=log_file, epoch_key=epoch_key,
                                 loss_key=loss_key, keep=lambda x: x[:12] == '[Validating]')
    epochs = train.keys()

    padding = 15
    print('{} | {} | {}'.format('EPOCH'.rjust(padding // 3),
                                'TRAIN LOSS'.rjust(padding),
                                'VALIDATION LOSS'.rjust(padding)))
    for epoch in epochs:
        train_loss = '{:.7f}'.format(float(train[epoch]['average'])).rjust(padding)
        try:
            valid_loss = '{:.7f}'.format(float(valid[epoch]['average'])).rjust(padding)
        except KeyError:
            valid_loss = '---'
        print('{} | {} | {}'.format('{}'.format(epoch).rjust(padding // 3), train_loss, valid_loss))
