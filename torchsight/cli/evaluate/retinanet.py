"""Evaluate the RetinaNet model."""
import click


@click.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True))
@click.option('-d', '--dataset', default='coco', type=click.Choice(['coco', 'flickr32']),
              help='The dataset to use to validate.', show_default=True)
@click.option('-dr', '--dataset-root', help='The root directory where is the data of the dataset.')
@click.option('--results-dir', default='.', show_default=True, help='The directory where to store the results.')
@click.option('--subdataset', help='The subdataset to use in the dataset (val2017, 25_brands_test.txt, etc).')
@click.option('--batch-size', default=8, show_default=True)
@click.option('--num-workers', default=8, show_default=True)
@click.option('--device', help='The device where to run the evaluation. Default to cuda:0 if cuda is available.')
@click.option('--threshold', default=0.5, help='The detection threshold.', show_default=True)
@click.option('--iou-threshold', default=0.5, help='The IoU threshold for the NMS.', show_default=True)
def retinanet(checkpoint, dataset, dataset_root, results_dir, subdataset,
              batch_size, num_workers, device, threshold, iou_threshold):
    """Evaluate the RetinaNet model with the indicated dataset that contains its data in DATASET-ROOT with the
    model saved at CHECKPOINT."""
    import torch

    from torchsight.evaluators.retinanet import RetinaNetFlickr32Evaluator

    device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if dataset == 'flickr32':
        RetinaNetFlickr32Evaluator(
            checkpoint,
            params={
                'root': dataset_root,
                'file': '{}/flickr32_predictions.csv'.format(results_dir),
                'dataset': subdataset if subdataset is not None else 'test',
                'dataloader': {
                    'num_workers': num_workers,
                    'batch_size': batch_size
                },
                'iou_threshold': iou_threshold,
                'thresholds': {
                    'detection': threshold,
                    'iou': iou_threshold
                }
            },
            device=device
        ).evaluate()
    else:
        raise NotImplementedError()
