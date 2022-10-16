from dataclasses import dataclass

import torch


def split(imgs):
    *_, height, width = imgs.shape
    left, right = imgs[..., :, : width // 2], imgs[..., :, width // 2 :]
    return left, right


@dataclass
class ShuffledBatch:
    lefts: torch.Tensor
    rights: torch.Tensor
    idxs_lefts: torch.Tensor
    idxs_rights: torch.Tensor
    labels: torch.Tensor


def create_shuffled_batch(mnist_batch, frac2shuffle=0.5):
    imgs, _ = mnist_batch
    bs, *_ = imgs.shape

    # randomly subset some digits for which the left will be
    # paired with a different right; this is also the label!
    m = frac2shuffle < torch.FloatTensor(bs).uniform_()
    num2shuffle = m.sum()
    permutation_idxs = torch.randperm(num2shuffle)

    # if any index is __not__ permuted, randomly permute again
    if 0 < (permutation_idxs == torch.arange(num2shuffle)).sum():
        return create_shuffled_batch(mnist_batch, frac2shuffle)

    # split shuffled from unshuffled and shuffle the left sides
    unshuffled_lefts, unshuffled_rights = split(imgs[~m, 0, ...])
    shuffled = imgs[m, 0, ...]
    shuffled_lefts, _ = split(shuffled[permutation_idxs, ...])
    _, shuffled_rights = split(shuffled)

    # combine the shuffled and unshuffled lefts and rights
    lefts = torch.concat([unshuffled_lefts, shuffled_lefts], axis=0)
    rights = torch.concat([unshuffled_rights, shuffled_rights], axis=0)

    # flip the label so that 1.0 is a match
    labels, _ = torch.sort(m.float().to(imgs.device))
    labels = 1.0 - labels

    # re-order in terms of original bottom order
    permuted_idxs = torch.arange(bs)[m]
    unpermuted_idxs = torch.arange(bs)[~m]
    idxs_lefts = torch.concat([unpermuted_idxs, permuted_idxs[permutation_idxs]])
    idxs_rights = torch.concat([unpermuted_idxs, permuted_idxs])

    _, order = torch.sort(idxs_rights)
    idxs_lefts = idxs_lefts[order]
    idxs_rights = idxs_rights[order]
    lefts = lefts[order, ...]
    rights = rights[order, ...]
    labels = labels[order]

    return ShuffledBatch(
        lefts=lefts.unsqueeze(1),
        rights=rights.unsqueeze(1),
        idxs_lefts=idxs_lefts,
        idxs_rights=idxs_rights,
        labels=labels,
    )
