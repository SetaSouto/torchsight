"""Script to show how to use a Trainer."""
from torchsight.trainers import RetinaNetTrainer

RetinaNetTrainer(
    hyperparameters={
        'RetinaNet': {'classes': 1, 'resnet': 18},
        'datasets': {'class_names': ('person',), 'train': 'val2017', 'validation': 'val2017'},
        'optimizer': {'learning_rate': 1e-3}
    },
    logs='./logs/retinanet',
    checkpoint='./logs/retinanet/1548542842/checkpoint_epoch_1.pth.tar'
).train()
