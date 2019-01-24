"""Initialize the package.

Import directly to perform more beautiful imports.

See:
http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html
"""
from .anchors import Anchors
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet
from .retinanet import RetinaNet
