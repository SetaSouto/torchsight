"""Commands to interpolate the sizes of some structures."""
import click


@click.command()
@click.option('--image-size', default=512, show_default=True)
@click.option('--images', default=12000, show_default=True)
@click.option('--strides', default='8 16 32 64 128', show_default=True)
@click.option('--anchors-per-loc', default=9, show_default=True)
@click.option('--embedding-size', default=256, show_default=True)
@click.option('--keep-per-neigh', default=9, show_default=True,
              help='Of the total anchors per neighborhood, how many of them must be saved.')
@click.option('--neigh', default=1, show_default=True,
              help='Neighborhood size based on the locations in the feature map.')
def sizes(image_size, images, strides, anchors_per_loc, embedding_size, keep_per_neigh, neigh):
    """Compute the number of embeddings per image, total number of embeddings of the dataset,
    the size of the dict to map between embedding id and image file, and the size of the
    approximate size of the tensor that holds the embeddings and the other that holds
    the bounding boxes.
    """
    import sys

    import torch

    def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    # Compute the total number of anchors for the dataset
    num_anchors = 0
    keep_per_neigh = min(keep_per_neigh, anchors_per_loc * (neigh ** 2))
    for stride in (int(s) for s in strides.split()):
        side = int(image_size / stride)
        locations = side ** 2
        print('With stride {}:'.format(stride))
        print('  - Locations in the Feature Map: {} ({side} x {side})'.format(locations, side=side))
        print('  - Activation zone size:         {stride} x {stride}'.format(stride=stride))
        num_anchors += ((image_size / stride / neigh) ** 2) * keep_per_neigh
    num_anchors = int(num_anchors)
    total_anchors = num_anchors * images

    print('\nAnchors per image: {}'.format(human_format(num_anchors)))
    print('Total anchors:     {}'.format(human_format(total_anchors)))

    path = '/dir/images/foo_image_path.jpg'
    dict_size = num_anchors
    id_to_image = {i: path for i in range(int(dict_size))}
    print('Dict size:         {}'.format(sizeof_fmt(sys.getsizeof(id_to_image) * images)))

    tensor = torch.rand(embedding_size)
    print('Tensors size:      {}'.format(sizeof_fmt(tensor.element_size() * tensor.nelement() * total_anchors)))
