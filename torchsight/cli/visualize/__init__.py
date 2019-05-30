"""Commands to visualize the datasets or the results."""
import click

from .dldenet import dldenet
from .logo32plus import logo32plus
from .sigmoid import sigmoid


@click.group()
def visualize():
    """Visualize command to see datasets and results."""


visualize.add_command(dldenet)
visualize.add_command(logo32plus)
visualize.add_command(sigmoid)
