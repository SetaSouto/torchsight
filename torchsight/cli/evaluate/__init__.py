"""Commands to evaluate the models."""
import click

from .dldenet import dldenet
from .retinanet import retinanet


@click.group()
def evaluate():
    """Evaluate the different models."""


evaluate.add_command(dldenet)
evaluate.add_command(retinanet)
