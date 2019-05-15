"""Commands to evaluate the models."""
import click

from .dldenet import dldenet


@click.group()
def evaluate():
    """Evaluate the different models."""


evaluate.add_command(dldenet)
