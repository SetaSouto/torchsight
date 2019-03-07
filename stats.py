"""Script to show some stats on demand."""
from torchsight.loggers import Logger

LOGGER = Logger(description=None, directory='/media/souto/DATA/HDD/logs/retinanet/1551992798', filename='logs.json')
# print(LOGGER.average_loss(key='Total', window=406))
print(LOGGER.epoch_losses(epoch_key='Epoch', loss_key='Total'))
