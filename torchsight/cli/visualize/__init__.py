"""Commands to visualize the datasets or the results."""
import click

from .coco import coco
from .dldenet import dldenet
from .flickr32 import flickr32
from .instre import instre
from .logo32plus import logo32plus
from .modanet import modanet
from .retinanet import retinanet
from .sigmoid import sigmoid


@click.group()
def visualize():
    """Visualize command to see datasets and results."""


visualize.add_command(coco)
visualize.add_command(dldenet)
visualize.add_command(flickr32)
visualize.add_command(instre)
visualize.add_command(logo32plus)
visualize.add_command(modanet)
visualize.add_command(retinanet)
visualize.add_command(sigmoid)
