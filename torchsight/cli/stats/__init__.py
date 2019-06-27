"""Commands to extract some stats from the log files."""
import click

from .boxes import describe_boxes
from .dlde import dlde
from .print import printlogger
from .sizes import sizes


@click.group()
def stats():
    """Commands to compute statistics or metrics"""


stats.add_command(describe_boxes)
stats.add_command(dlde)
stats.add_command(printlogger)
stats.add_command(sizes)
