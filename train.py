"""Script to show how to use a Trainer."""
from torchsight.trainers import RetinaNetTrainer

RetinaNetTrainer(
    hyperparameters={
        'optimizer': {'learning_rate': 0.001},
        'RetinaNet': {'classes': 2},
        'datasets': {'class_names': ('airplane', 'kite'), 'train': 'val2017'}
    },
    logs_dir='/media/souto/DATA/HDD/logs/retinanet',
    checkpoint='/media/souto/DATA/HDD/logs/retinanet/1551992798/checkpoint_epoch_XX.pth.tar'
).train(validate=False)

# TODO: Create directional module
# Title: Torchsight, a novel framework for few-shot learning using deep local directional embeddings (DLDE).
# TODO: Try to train a single detector for humans. At least try to overfit a model, to test that it can learn.

# Ok, it seems that is learning. We must amplify a lot the classification and train for a lot of epochs.
# TODO: Evaluation, train with the airplane and kite samples but with the training set and make the evaluation
# module. Check the mAP function because we have to calculate the mAP per class, read the COCO evaluation
# and check some implementation. While doing this overfit the network.
