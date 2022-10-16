import itertools
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet18
from tqdm.auto import tqdm

from amphisbaena.data import create_shuffled_batch


@dataclass
class LeftRightAssignment:
    idxs_left_pred: np.array
    similarity_matrix: np.array


class Amphisbaena(LightningModule):
    def __init__(self, frac2shuffle=0.5, lr=1e-3, backbone="lin"):
        """
        Args:
            frac2shuffle (float): during training, the fraction of training
                examples for which the left is shown to the model with its
                corresponding right
            lr (float): learning rate
            backbone ("lin" | "conv" | "conv_pretrainer"): the type of image
                recognition arm
        """
        super().__init__()

        self.frac2shuffle = frac2shuffle
        self.lr = lr
        self.backbone = backbone
        self.save_hyperparameters()

        self.left = self.arm(backbone, 512)
        self.right = self.arm(backbone, 512)
        self.similarity = nn.Linear(512, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def arm(self, backbone, n_outputs):
        if backbone.startswith("conv"):
            pretrained = backbone == "conv_pretrained"
            # 1000 is the number of imagenet classes
            arm_ = resnet18(
                num_classes=1000,
                pretrained=pretrained,
            )
            arm_.fc = nn.Linear(
                in_features=512,
                out_features=n_outputs,
                bias=True,
            )
            return arm_
        elif backbone == "lin":
            return nn.Sequential(
                nn.Linear(28 * 14, 512),
                nn.ReLU(),
                nn.Linear(512, n_outputs),
                nn.ReLU(),
            )
        else:
            raise ValueError

    def forward(self, lefts, rights, probability=False):
        bs, _, height, width = lefts.shape
        if self.backbone == "lin":
            lefts = lefts.reshape(bs, height * width)
            rights = rights.reshape(bs, height * width)
        elif self.backbone.startswith("conv"):
            lefts = lefts.repeat(1, 3, 1, 1)
            rights = rights.repeat(1, 3, 1, 1)
        else:
            raise ValueError
        left_latents = self.left(lefts)
        right_latents = self.right(rights)
        logits = self.similarity(torch.abs(left_latents - right_latents))

        return torch.sigmoid(logits) if probability else logits

    def assign(self, lefts, rights, bs=None) -> LeftRightAssignment:
        """Assign lefts to rights

        Args:
            lefts (bs x 1 x 14 x 28 tensor): digit lefts
            rights (bs x 1 x 14 x 28 tensor): digit rights
            bs (int, optional): batch size for inference

        Returns:
            leftrightAssignment: left-right assignment
        """

        if bs is None:
            bs = 512 if torch.cuda.is_available() else 64
        n, *_ = lefts.shape
        S = np.zeros((n, n))

        combinations = []
        iter_ = itertools.product(enumerate(rights), enumerate(lefts))
        for (i, right), (j, left) in tqdm(list(iter_), unit="pair"):
            combinations.append((i, j, left.unsqueeze(0), right.unsqueeze(0)))

        for batch_idx in tqdm(range(0, len(combinations), bs), unit="batch"):
            batch = combinations[batch_idx : batch_idx + bs]
            _is, _js, lefts, rights = zip(*batch)
            left = torch.concat(lefts, axis=0)
            right = torch.concat(rights, axis=0)
            with torch.no_grad():
                similarities = self(left, right, probability=True)
            for i, j, similarity in zip(_is, _js, similarities):
                S[i, j] = similarity.item()

        _, idxs_left_pred = linear_sum_assignment(S, maximize=True)

        return LeftRightAssignment(
            idxs_left_pred=idxs_left_pred,
            similarity_matrix=S,
        )

    def step(self, batch):
        batch = create_shuffled_batch(batch, self.frac2shuffle)
        preds = self(batch.lefts, batch.rights).squeeze()
        loss = self.criterion(preds, batch.labels).sum()
        return loss

    def training_step(self, batch):
        loss = self.step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss = self.step(batch)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
