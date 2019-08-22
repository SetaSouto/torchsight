"""Commands for the retrievers experiments."""
import click

from .flickr32 import flickr32


@click.group()
def retrievers():
    """Commands for the retrivers experiments."""


retrievers.add_command(flickr32)
