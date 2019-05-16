"""Package where are the CLI commands."""
import click

from .evaluate import evaluate
from .stats import stats
from .train import train
from .visualize import visualize


@click.group()
def cli():
    """Command Line Interface to interact with the project."""


# COMMANDS OF THE CLI
cli.add_command(evaluate)
cli.add_command(stats)
cli.add_command(train)
cli.add_command(visualize)
