"""Commands to visualize the COCO dataset."""
import click


@click.command()
@click.option('-dr', '--dataset_root', required=True)
def coco(dataset_root):
    """Visualize images randomly of the COCO dataset."""
    import random
    from torchsight.datasets.coco import CocoDataset
    from torchsight.transforms.augmentation import AugmentDetection
    from torchsight.utils.visualize import visualize_boxes

    dataset = CocoDataset(root=dataset_root, dataset='train2017', transform=AugmentDetection())

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in indices:
        image, boxes, *_ = dataset[i]
        visualize_boxes(image, boxes, label_to_name=dataset.classes['names'])
