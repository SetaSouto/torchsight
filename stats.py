"""Script to show some stats on demand."""
from torchsight.loggers import Logger

LOGGER = Logger(description=None, directory='./logs/retinanet/1548681009', filename='logs.txt')
print(LOGGER.average_loss(key='Total', window=1e4))
