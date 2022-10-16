import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback
from torchvision import transforms as T

import wandb
from amphisbaena.data import create_shuffled_batch, split
from amphisbaena.metrics import compute_accuracy


def visualize_shuffled_batch(batch):
    X, _ = batch
    batch_shuffled = create_shuffled_batch(batch, 0.5)
    bs, *_ = X.shape
    fig, axes = plt.subplots(2, bs, figsize=(bs, 2))
    iter_ = zip(batch_shuffled.lefts, batch_shuffled.rights, batch_shuffled.labels, X)
    for i, (left, right, label, img) in enumerate(iter_):
        canvas = np.zeros((28, 28))
        canvas[:, :14], canvas[:, 14:] = left, right
        axes[0, i].imshow(canvas, cmap="Blues" if label.item() else "Reds")
        axes[1, i].imshow(img.squeeze())
    return fig


def color2greyscale(imgs):
    color2greyscale_ = [
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((28, 28)),
        T.ToTensor(),
    ]
    color2greyscale_ = T.Compose(color2greyscale_)
    imgs_ = [color2greyscale_(img).unsqueeze(0) for img in imgs]
    return torch.concat(imgs_, axis=0)


def visualize_model_outputs(model, imgs):
    n_imgs, *_ = imgs.shape

    res = compute_accuracy(model, *split(imgs))

    # normalize between 0 and 1
    imgs -= imgs.min()
    imgs /= imgs.max()

    if model.backbone.startswith("conv"):
        imgs = color2greyscale(imgs)

    lefts, rights = split(imgs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    canvas = np.zeros((n_imgs * 28, n_imgs * 28))
    for i, left in enumerate(lefts):
        for j, right in enumerate(rights):
            left, right = left.cpu(), right.cpu()
            s = res.similarity_matrix[i, j]
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j * 28) + 14] = left * s
            canvas[i * 28 : (i + 1) * 28, (j * 28) + 14 : (j + 1) * 28] = right * s
    canvas[canvas == 0] = -1
    m = ax.imshow(canvas, cmap="RdYlBu")
    fig.colorbar(m)
    fig.suptitle(f"Overall Accuracy: {res.accuracy:.1%}")
    return fig


class VisualizeAssignmentCallback(Callback):
    def __init__(self):
        self.prev_batch = None

    def on_validation_batch_end(self, _a, _b, _c, batch, _d, _e):
        self.prev_batch = batch

    def on_validation_epoch_end(self, _, pl_module):
        (imgs, _) = self.prev_batch
        imgs = imgs[:8, ...]
        fig = visualize_model_outputs(pl_module, imgs)
        wandb.log({"assignment": fig})
        plt.close()
