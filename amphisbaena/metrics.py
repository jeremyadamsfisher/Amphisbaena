import os
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from pytorch_lightning import Callback

import wandb
from amphisbaena.data import split
from amphisbaena.model import LeftRightAssignment


@dataclass
class Accuracy(LeftRightAssignment):
    accuracy: float


def compute_accuracy(model, lefts, rights) -> Accuracy:
    n, *_ = lefts.shape
    assignment = model.assign(lefts, rights)
    idxs_left_true = np.arange(n)
    n_correct = (idxs_left_true == assignment.idxs_left_pred).sum()
    if (assignment.similarity_matrix == assignment.similarity_matrix[0, 0]).all():
        logger.debug(
            "encountered corner case where every element of the similarity "
            "matrix is the same, so the bipartite match solver outputs "
            "0..n (where n is the rank of the matrix), but also happens to "
            "be the 'ground truth' label. "
        )
        n_correct = 0
    return Accuracy(
        accuracy=n_correct / n,
        idxs_left_pred=assignment.idxs_left_pred,
        similarity_matrix=assignment.similarity_matrix,
    )


class ComputeAccuracyCallback(Callback):
    def __init__(self, save_most_accurate: bool = True):
        """Compute the model left-right matching accuracy, optionally
        saving the highest performing model checkpoint

        Args:
            save_most_accurate (bool, optional): log the highest performing model
                to wandb. Defaults to False.
        """
        self.batches = []
        self.accuracy = None
        self.accuracy_highest = float("-inf")
        self.save_most_accurate = save_most_accurate
        self.checkpoint_fname = f"{uuid.uuid4()}.ckpt"

    def on_validation_batch_end(self, _a, _b, _c, batch, _d, _e):
        imgs, _ = batch
        self.batches.append(imgs)

    def on_validation_epoch_end(self, trainer, pl_module):
        validation_imgs = torch.concat(self.batches, axis=0)
        accuracy = compute_accuracy(pl_module, *split(validation_imgs)).accuracy
        self.log("val/accuracy", accuracy)

        if self.accuracy_highest < accuracy and self.save_most_accurate:
            logger.info("logging most accurate model so far: {:.2%}", accuracy)
            trainer.save_checkpoint(os.path.join(wandb.run.dir, self.checkpoint_fname))
            self.accuracy_highest = accuracy

        self.accuracy = accuracy
        self.batches = []
