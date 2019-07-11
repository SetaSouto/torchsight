"""Commands for the experiments."""
import click

from .flickr32 import flickr32


@click.group()
def experiments():
    """Commands for the experiments."""


experiments.add_command(flickr32)
