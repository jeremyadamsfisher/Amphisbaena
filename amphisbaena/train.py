import multiprocessing
import pathlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from torchvision import transforms as T

from amphisbaena.metrics import ComputeAccuracyCallback
from amphisbaena.model import Amphisbaena
from amphisbaena.viz import VisualizeAssignmentCallback


@dataclass
class TrainingOutputs:
    model: nn.Module
    trainer: Trainer
    data: LightningDataModule
    accuracy: float
    checkpoint_most_accurate: Optional[pathlib.Path]


def train(
    max_epochs=10,
    lr=1e-2,
    backbone="lin",
    trainer_config=None,
    save_most_accurate=True,
    checkpoint_fp=None,
) -> TrainingOutputs:
    """Train amphisbaena

    Args:
        max_epochs (int, optional): Maximum training epochs. Defaults to 10.
        lr (_type_, optional): Learning rate. Defaults to 1e-2.
        backbone (str, optional): Image recognition arm type. Defaults to "lin".
        trainer_config (_type_, optional): Lightning Trainer configuration. Defaults to None.
        save_most_accurate (boolean, optional): If True, save the most accurate checkpoint. Defaults to True.
        checkpoint_fp (str, optional): Optionally restore from checkpoint. Defaults to None.

    Returns:
        TrainingOutputs: training outputs
    """
    if trainer_config is None:
        trainer_config = {}
    seed_everything(42, workers=True)
    data = MNISTDataModule(
        num_workers=multiprocessing.cpu_count(),
        normalize=True,
        batch_size=512 if torch.cuda.is_available() else 64,
        val_split=0.01,
    )
    if backbone.startswith("conv"):
        pipeline = [T.Resize((100, 100)), data.default_transforms()]
        transforms = T.Compose(pipeline)
        data.train_transforms = data.val_transforms = data.test_transforms = transforms
    else:
        data.train_transforms = data.val_transforms = data.test_transforms = None
    data.prepare_data()
    data.setup()
    compute_accuracy_callback = ComputeAccuracyCallback(
        save_most_accurate=save_most_accurate
    )
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[compute_accuracy_callback, VisualizeAssignmentCallback()],
        deterministic=True,
        max_epochs=max_epochs,
        **trainer_config
    )
    if checkpoint_fp:
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, _: storage)
        hyperparameters = checkpoint["hyper_parameters"]
        model = Amphisbaena(**hyperparameters)
        trainer.fit(model, data, ckpt_path=checkpoint_fp)
    else:
        model = Amphisbaena(lr=lr, backbone=backbone)
        trainer.fit(model, data)
    return TrainingOutputs(
        model=model,
        trainer=trainer,
        data=data,
        accuracy=compute_accuracy_callback.accuracy,
        checkpoint_most_accurate=compute_accuracy_callback.checkpoint_fname
        if save_most_accurate
        else None,
    )