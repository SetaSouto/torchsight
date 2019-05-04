"""Commands to extract some stats from the log files."""
import click

from .print import printlogger


@click.group()
def stats():
    """Extract stats from the logs."""


stats.add_command(printlogger)
