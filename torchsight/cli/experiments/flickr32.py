"""Commands for the experiments using the Flickr32 dataset."""
import click


@click.group()
def flickr32():
    """Commands for the experiments using the Flickr32 dataset."""


def load_config(config):
    """Load the json config.

    Arguments:
        config (str): the path to the config file.

    Returns:
        dict: the loaded config.
    """
    import json

    with open(config, 'r') as file:
        return json.loads(file.read())


@flickr32.command()
@click.option('-c', '--config', required=True, type=click.Path(exists=True),
              help='The configuration to use as params in the experiment.')
@click.option('--device', help='The device to use to run the experiment.')
def run(config, device):
    """Run the experiment with the given configuration."""
    from torchsight.experiments.retrievers.flickr32.experiment import Flickr32RetrieverExperiment

    return Flickr32RetrieverExperiment(params=load_config(config), device=device).run()


@flickr32.command()
@click.option('-c', '--config', required=True, type=click.Path(exists=True), help='The config used for the experiment.')
@click.option('-b', '--brand', help='Show the results only for this single brand.')
@click.option('-k', default=10, show_default=True, help='The number of results to see.')
@click.option('--queries', is_flag=True, help='Visualize only the queries instead of the queries and their results.')
def visualize(config, brand, k, queries):
    """Visualize the results of the experiment."""
    from torchsight.experiments.retrievers.flickr32.experiment import Flickr32RetrieverExperiment

    experiment = Flickr32RetrieverExperiment(params=load_config(config))

    if queries:
        return experiment.visualize_queries()

    return experiment.visualize_results(brand, k)


@flickr32.command()
@click.option('-r', '--root', required=True, help='The root directory where is the data of the dataset. '
              'See Flickr32 dataset for more docs.')
@click.option('-k', default=27, show_default=True, help='The number of brands to select.')
@click.option('-d', '--directory', default='./torchsight/experiments/few-shot/flickr32/', show_default=True,
              help='The directory where to store the generated files.', type=click.Path(exists=True))
def random_select_brands(root, k, directory):
    """Random select the brands that will be used for training and for the experiment and save them
    in the given directory as 'train.csv' and 'eval.csv'.
    """
    import os
    import random

    from torchsight.datasets import Flickr32Dataset

    brands = [b for b in Flickr32Dataset(root).brands if b != 'no-logo']
    train = random.sample(brands, k)
    evaluation = [b for b in brands if b not in train]

    with open(os.path.join(directory, 'train.csv'), 'w') as file:
        file.write('\n'.join(train))

    with open(os.path.join(directory, 'eval.csv'), 'w') as file:
        file.write('\n'.join(evaluation))


@flickr32.command()
@click.option('-c', '--config-file', default='./torchsight/experiments/few-shot/flickr32/config.json',
              type=click.Path(exists=True), help='The base configuration for the training.')
@click.option('-b', '--brands-file', default='./torchsight/experiments/few-shot/flickr32/train.csv', show_default=True,
              type=click.Path(exists=True), help='The file with the brands for training.')
def set_brands_in_config(config_file, brands_file):
    """Given a configuration for the training, this commands sets the brands to use in the training."""
    import json

    with open(config_file, 'r') as file:
        config = json.loads(file.read())

    with open(brands_file, 'r') as file:
        brands = file.read().split('\n')

    config['datasets']['use'] = 'flickr32'
    config['datasets']['flickr32']['brands'] = brands

    with open(config_file, 'w') as file:
        file.write(json.dumps(config, indent=2))
