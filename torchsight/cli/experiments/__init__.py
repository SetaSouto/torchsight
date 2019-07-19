"""Commands for the experiments."""
import click

from .flickr32 import flickr32


@click.group()
def experiments():
    """Commands for the experiments."""


@click.group()
def retrievers():
    """Commands for the retrivers experiments."""


retrievers.add_command(flickr32)

experiments.add_command(retrievers)
