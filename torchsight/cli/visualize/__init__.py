"""Commands to visualize the datasets or the results."""
import click

from .logo32plus import logo32plus


@click.group()
def visualize():
    """Visualize command to see datasets and results."""


visualize.add_command(logo32plus)
