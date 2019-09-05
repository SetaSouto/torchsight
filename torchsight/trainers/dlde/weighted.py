"""DLDENet trainer for the weighted version."""
import torch

from torchsight.losses import DLDENetLoss
from torchsight.models import DLDENet

from ..retinanet import RetinaNetTrainer


class DLDENetTrainer(RetinaNetTrainer):
    """Deep Local Directional Embedding with tracked means trainer.

    As the architecture is very similar with RetinaNet we use the same trainer and only
    override some attributes and methods. For more information please read the RetinaNet
    trainer documentation.
    """

    ####################################
    ###           GETTERS            ###
    ####################################

    @staticmethod
    def get_base_hp():
        """Get the base hyperparameters of the trainer.

        Returns:
            JsonObject: with the base hyperparameters.
        """
        return RetinaNetTrainer.get_base_hp().merge({
            'model': {
                # If you provide a checkpoint all the other hyperparameters will be overriden
                # by the ones in the checkpoint and the model will be initialized with the
                # weights of the checkpoint except for the classification layer (to work with
                # a possible mismatching number of classes)
                'checkpoint': None,
                # If you set this flag to True the trainer will run an epoch to get the "mean
                # embedding" of each class and will set that embedding as the classification
                # weight of the classification head
                'init_classification_weights': False,
                'init_classification_weights_norm': 4,
                # The hyperparameters of the model if no checkpoint is provided
                'resnet': 18,
                'features': {
                    'pyramid': 256,
                    'regression': 256,
                    'classification': 256
                },
                'anchors': {
                    'sizes': [32, 64, 128, 256, 512],
                    'scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)],
                    'ratios': [0.5, 1, 2]
                },
                'fpn_levels': [3, 4, 5, 6, 7],
                'embedding_size': 256,
                'normalize': True,
                'pretrained': True,
                'evaluation': {'threshold': 0.5, 'iou_threshold': 0.5},
                'weighted_bias': False,
                'fixed_bias': 0,
                'increase_norm_by': None
            },
            'criterion': {
                'alpha': 0.25,
                'gamma': 2.0,
                'sigma': 3.0,
                'iou_thresholds': {'background': 0.4, 'object': 0.5},
                'increase_foreground_by': 100,
                # Weight of each loss. See train method.
                'weights': {'classification': 100, 'regression': 1, 'similarity': 1, 'dispersion': 1},
                'soft': False,  # Apply soft loss weighted by the IoU
            },
            'datasets': {
                'use': 'coco',
                'coco': {
                    'root': './datasets/coco',
                    'class_names': (),  # () indicates all classes
                    'train': 'train2017',
                    'validation': 'val2017'
                },
                'logo32plus': {
                    'root': './datasets/logo32plus',
                    'classes': None,
                },
                'flickr32': {
                    'root': './datasets/flickr32',
                    'brands': None,
                    'only_boxes': True,
                    'training': 'trainval',  # The name of the dataset to use for training
                    'validation': 'test'  # The name of the dataset to use for validation
                }
            },
            'dataloaders': {
                'batch_size': 1,
                'shuffle': True,
                'num_workers': 8
            },
            'optimizer': {
                'use': 'sgd',  # Which optimizer the trainer must use
                'adabound': {
                    'lr': 1e-3,  # Learning rate
                    'final_lr': 1  # When the optimizer change from Adam to SGD
                },
                # Lookahead is not an optimizer by itself, you must choose another of
                # the optimizers and set the 'use' flag to True in the lookahead params
                'lookahead': {
                    'use': False,
                    'k': 5,
                    'alpha': 0.5
                },
                'radam': {
                    'lr': 1e-3,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 0
                },
                'sgd': {
                    'lr': 1e-2,
                    'momentum': 0.9,
                    'weight_decay': 1e-4
                }
            },
            'scheduler': {
                'factor': 0.1,
                'patience': 5,
                'threshold': 0.01
            },
            'transform': {
                'GaussNoise': {
                    'var_limit': (10, 50),
                    'p': 0.5
                },
                'GaussianBlur': {
                    'blur_limit': 0.7,
                    'p': 0.5
                },
                'RandomBrightnessContrast': {
                    'brightness_limit': 0.2,
                    'contrast_limit': 0.2,
                    'p': 0.5
                },
                'Rotate': {
                    'limit': 45,
                    'p': 0.5
                },
                'LongestMaxSize': {
                    'max_size': 512
                },
                'PadIfNeeded': {
                    'min_height': 512,
                    'min_width': 512
                },
                'RandomSizedBBoxSafeCrop': {
                    'height': 512,
                    'width': 512
                }
            }
        })

    def get_model(self):
        """Initialize and get a DLDENet model instance."""
        hyperparameters = self.hyperparameters.model.dict()

        # Load the model from the checkpoint
        if hyperparameters['checkpoint'] is not None:
            checkpoint = torch.load(hyperparameters['checkpoint'], map_location=self.device)
            self.hyperparameters.model.merge(checkpoint['hyperparameters']['model'])
            self.hyperparameters.model.checkpoint = hyperparameters['checkpoint']

            return DLDENet.from_checkpoint_with_new_classes(
                checkpoint=checkpoint,
                num_classes=hyperparameters['classes'],
                device=self.device
            )

        # No checkpoint detected, start only with the weights of the ResNet
        return DLDENet(
            classes=hyperparameters['classes'],
            resnet=hyperparameters['resnet'],
            features=hyperparameters['features'],
            anchors=hyperparameters['anchors'],
            fpn_levels=hyperparameters['fpn_levels'],
            embedding_size=hyperparameters['embedding_size'],
            normalize=hyperparameters['normalize'],
            pretrained=hyperparameters['pretrained'],
            device=self.device,
            weighted_bias=hyperparameters['weighted_bias'],
            fixed_bias=hyperparameters['fixed_bias'],
            increase_norm_by=hyperparameters['increase_norm_by'],
        )

    def get_criterion(self):
        """Get the criterion to use to train the model.

        Returns:
            DLDENetLoss: The unified loss between FocalLoss and Cosine similarity Loss.
        """
        params = self.hyperparameters.criterion.dict()

        return DLDENetLoss(
            alpha=params['alpha'],
            gamma=params['gamma'],
            sigma=params['sigma'],
            iou_thresholds=params['iou_thresholds'],
            increase_foreground_by=params['increase_foreground_by'],
            soft=params['soft'],
            device=self.device
        )

    ####################################
    ###           METHODS            ###
    ####################################

    def before_train_callback(self):
        """A method that is called before the training starts."""
        if self.hyperparameters.model.init_classification_weights:
            self.init_classification_weights()

    def init_classification_weights(self):
        """Init the classification weights of the model with the "mean embedding" of each class.

        We'll compute the embedding for each object in the dataset and accumulate it in a tensor
        that will be the initial weights of the classification head.
        """
        print('Initializing classification weights with mean embeddings')
        with torch.no_grad():
            # Start with zero weights
            classes = self.hyperparameters.model.classes
            embedding_size = self.hyperparameters.model.embedding_size
            weights = torch.zeros(embedding_size, classes).to(self.device)

            # Move the model to the correct device
            self.model.to(self.device)
            n_batches = len(self.dataloader)

            for i, (images, annotations, *_) in enumerate(self.dataloader):
                images, annotations = images.to(self.device), annotations.to(self.device)

                # Compute embeddings and anchors
                feat_maps = self.model.fpn(images)
                embeddings = torch.cat([self.model.classification.encode(fm) for fm in feat_maps], dim=1)   # (b, e, d)
                anchors = self.model.anchors(images)                                                        # (b, e, 4)

                # Iterate over the annotations assign them and get the embeddings whose anchors has IoU > threshold
                thresholds = self.hyperparameters.criterion.iou_thresholds.dict()
                for j, item_annotations in enumerate(annotations):
                    item_anchors = anchors[j]
                    assignations = self.model.anchors.assign(item_anchors, item_annotations, thresholds)
                    assigned_annot, object_mask, *_ = assignations              # (e , 5), (e,)
                    item_embds = embeddings[j, object_mask]                     # (e', d)
                    item_embds_labels = assigned_annot[object_mask, 4].long()   # (e',)
                    for k, label in enumerate(item_embds_labels):
                        weights[:, label] += item_embds[k]

                self.logger.log({
                    'Initializing': None,
                    'Batch': '{}/{}'.format(i+1, n_batches)
                })

            # Normalize the weights and scale them
            weights /= weights.norm(dim=0, keepdim=True)
            weights *= self.hyperparameters.model.init_classification_weights_norm

            # Assign this new init weights to the classification head
            self.model.classification.weights.data.copy_(weights)

    def forward(self, *args):
        """Forward pass through the network and loss computation.

        Returns:
            torch.Tensor: The loss of the batch.
        """
        images, annotations, *_ = args
        images, annotations = images.to(self.device), annotations.to(self.device)

        anchors, regressions, classifications = self.model(images)
        del images

        losses = self.criterion(anchors, regressions, classifications, annotations, self.model)
        del anchors, regressions, classifications, annotations

        classification, regression, similarity, dispersion = losses

        # Amplify the losses according to the criterion weights
        weights = self.hyperparameters['criterion']['weights']
        classification *= weights['classification']
        regression *= weights['regression']
        similarity *= weights['similarity']
        dispersion *= weights['dispersion']

        loss = classification + regression + similarity + dispersion

        # Log the classification and regression loss too:
        self.current_log['Class.'] = '{:.4f}'.format(float(classification))
        self.current_log['Pos'] = '{:.4f}'.format(float(self.criterion.focal.pos_loss * weights['classification']))
        self.current_log['Neg'] = '{:.4f}'.format(float(self.criterion.focal.neg_loss * weights['classification']))
        self.current_log['Regr.'] = '{:.4f}'.format(float(regression))
        self.current_log['Simil.'] = '{:.4f}'.format(float(similarity))
        self.current_log['Disp.'] = '{:4f}'.format(float(dispersion))
        # Log the mean norm of the weights in the classification module and their biases
        self.current_log['w-norm'] = '{:.4f}'.format(float(self.model.classification.weights.norm(dim=0).mean()))
        if self.model.classification.weighted_bias:
            self.current_log['bias'] = '{:.4f}'.format(float(self.model.classification.bias.mean()))

        return loss
