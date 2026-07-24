import random

import numpy as np
import torch

from src.training.autocast import seed_everything
from src.training.cutmix_mixup_aug import smooth_soft_targets


def test_seed_everything_covers_python_numpy_and_torch():
    seed_everything(17, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(1))
    seed_everything(17, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_label_smoothing_preserves_soft_target_distribution():
    targets = torch.tensor([[0.75, 0.25, 0.0, 0.0]])
    smoothed = smooth_soft_targets(targets, label_smoothing=0.2)

    assert torch.allclose(smoothed.sum(dim=1), torch.ones(1))
    assert torch.allclose(
        smoothed, torch.tensor([[0.65, 0.25, 0.05, 0.05]])
    )
