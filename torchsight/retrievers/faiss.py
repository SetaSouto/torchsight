"""Retrievers using FAISS as database."""
import os
import time

import faiss
import torch

from .retriever import InstanceRetriever


class FaissInstanceRetriever(InstanceRetriever):
    """A retriver that looks for instance of objects in a database of images.

    You must provide a model in the `get_model()` method.

    You can call the `create_database()` method to create the database,
    and then query instances of objects using the `query()` method.
    """

    def __init__(self, *args, storage='./databases', index='IndexFlatIP', **kwargs):
        """Initialize the retriever.

        Arguments:
            storage (str, optional): The path to the directory where to store the data.
            index (str, optional): The index of FAISS to use to store the embeddings.
                The default one is the FlatIP (Inner Product) that performs the cosine distance
                so your embeddings must be normalized beforehand.
                You can find more indexes here:
                https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

            The rest of the parameters are the same as InstanceRetriever.
        """
        self.database = None
        self.storage = storage
        self.embeddings_file = os.path.join(self.storage, 'embeddings.index')
        self.boxes_file = os.path.join(self.storage, 'boxes.index')
        self.index = index
        self.embeddings = None  # FAISS index for the embeddings
        self.boxes = None  # FAISS index for the boxes
        self.paths = {}  # A dict to map between FAISS ids and images' paths

    ##############################
    ###        SETTERS        ####
    ##############################

    def _set_indexes(self, dim):
        """Set the FAISS index.

        The embedding could have any size but the bounding boxes must have size 4.

        Arguments:
            dim (int): The dimension of the embeddings.
        """
        if self.index == 'IndexFlatL2':
            self.embeddings = faiss.IndexFlatL2(dim)
            self.boxes = faiss.IndexFlatL2(4)
        elif self.index == 'IndexFlatIP':
            self.embeddings = faiss.IndexFlatIP(dim)
            self.boxes = faiss.IndexFlatIP(4)
        else:
            raise ValueError('Index "{}" not supported.'.format(self.index))

    ######################################
    ###       DATABASE METHODS         ###
    ######################################

    def create_database(self, batch_size=8, num_workers=8):
        """Generates the database to insert in an index of FAISS.

        Arguments:
            batch_size (int): The batch size to use to compute in parallel the images.
            num_workers (int): The number of process to use to load the images and generate
                the batches.
        """
        self._print('Creating database ...')

        dataloader = self.dataset.get_dataloader(batch_size, num_workers)

        num_batches = len(dataloader)
        total_embs = 0
        total_imgs = 0
        init = time.time()

        with torch.no_grad():
            for i, (images, paths) in enumerate(dataloader):
                embeddings, boxes = self.model(images)

                # Create the indexes if they are not created yet
                if self.embeddings is None or self.boxes is None:
                    self._set_indexes(embeddings.shape[1])

                # Add the vectors to the indexes
                self.embeddings.add(embeddings)
                self.boxes.add(boxes)

                # Map the id of the vectors to their image path
                for j, path in paths:
                    self.paths[(i*batch_size) + j] = path

                # Show some stats about the progress
                total_imgs += images.shape[0]
                total_embs += embeddings.shape[0]
                self.logger.log({
                    'Batch': '{}/{}'.format(i + 1, num_batches),
                    'Time': '{:.3f} s'.format(time.time() - init),
                    'Images': total_imgs,
                    'Embeddings': total_embs,
                })

        self.save()

    def query(self, images, boxes=None, strategy='max_iou', k=100):
        """TODO:"""
        raise NotImplementedError()

    ###################################
    ###        SAVING/LOADING       ###
    ###################################

    def save(self):
        """Save the indexes in the storage directory."""
        self._print('Saving indexes ...')
        faiss.write_index(self.embeddings, self.embeddings_file)
        faiss.write_index(self.boxes, self.boxes_file)

    def load(self):
        """Load the indexes from the storage directory."""
        self._print('Loading indexes ...')

        if not os.path.exists(self.embeddings_file):
            raise ValueError('There is no ')

        self.embeddings = faiss.read_index(self.embeddings_file)
        self.boxes = faiss.read_index(self.boxes_file)
