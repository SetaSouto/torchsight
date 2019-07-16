"""Commands to extract some stats from the log files."""
import click

from .boxes import describe_boxes
from .dldenet import dldenet
from .focal import focal
from .print import printlogger
from .sizes import sizes


@click.group()
def stats():
    """Commands to compute statistics or metrics"""


stats.add_command(describe_boxes)
stats.add_command(dldenet)
stats.add_command(focal)
stats.add_command(printlogger)
stats.add_command(sizes)
