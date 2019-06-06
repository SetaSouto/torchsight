"""Utilities with the print function."""


class PrintMixin():
    """A mixin to print a namespace message with the name of the class."""

    def print(self, msg):
        """Print a namespaced message."""
        print('[{}] {}'.format(self.__class__.__name__, msg))
