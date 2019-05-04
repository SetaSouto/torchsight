"""Package where are the CLI commands."""
import click

from .stats import stats
from .train import train


@click.group()
def cli():
    """Command Line Interface to interact with the project."""


# COMMANDS OF THE CLI
cli.add_command(stats)
cli.add_command(train)
