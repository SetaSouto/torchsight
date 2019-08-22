"""Commands for the experiments."""
import click

from .few_shot import few_shot
from .retrievers import retrievers


@click.group()
def experiments():
    """Commands for the experiments."""


experiments.add_command(few_shot)
experiments.add_command(retrievers)
