"""Train commands for the CLI."""
import click

from .dldenet import dldenet, dldenet_from_checkpoint
from .retinanet import retinanet


@click.group()
def train():
    """Train a model with the given options."""


# COMMANDS OF THR GROUP
train.add_command(dldenet)
train.add_command(dldenet_from_checkpoint)
train.add_command(retinanet)
