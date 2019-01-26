"""Script to show how to use a Trainer."""
from torchsight.trainers import RetinaNetTrainer

RetinaNetTrainer(
    hyperparameters={
        'RetinaNet': {'classes': 1, 'resnet': 18},
        'datasets': {'class_names': ('person',)}
    },
    logs='./logs/retinanet'
).train()
