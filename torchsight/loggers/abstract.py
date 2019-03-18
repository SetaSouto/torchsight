"""Abstract logger to implement all the other loggers."""


class AbstractLogger():
    """Abstract logger class."""

    def log(self, data):
        """Log the data dict. You must implement this method on your logger class

        Arguments:
            data (dict): A dict with the data to log.
        """
        raise NotImplementedError()
