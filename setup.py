"""Setup to install and develop the package."""
from setuptools import setup, find_packages

setup(
    name="torchsight",
    packages=find_packages(exclude=('test',)),
    entry_points="""
        [console_scripts]
        torchsight=torchsight.cli:cli
    """
)
