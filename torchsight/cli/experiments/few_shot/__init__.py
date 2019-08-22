"""Commands for the few shot experiments."""
import click

from .flickr32 import flickr32


@click.group()
def few_shot():
    """Commands for the few shot experiments."""


few_shot.add_command(flickr32)
