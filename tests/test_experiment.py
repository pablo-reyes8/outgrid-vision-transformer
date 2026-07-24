from copy import deepcopy
from pathlib import Path

import torch

from src.experiment import load_experiment_config, prepare_experiment


def _synthetic_config(output_dir: Path) -> dict:
    return {
        "model": {
            "type": "model_a",
            "num_classes": 10,
            "stem_dim": 16,
            "dpr_max": 0.0,
            "stages": [{
                "dim": 16,
                "depth": 1,
                "num_heads": 4,
                "grid_size": 2,
                "outlook_heads": 4,
            }],
        },
        "training": {
            "epochs": 1,
            "optimizer": "adamw",
            "scheduler": "warmup_cosine",
            "mix_prob": 0.0,
            "label_smoothing": 0.0,
            "warmup_ratio": 0.0,
        },
        "data": {
            "dataset": "synthetic",
            "num_samples": 16,
            "batch_size": 4,
            "img_size": 8,
            "augmentation": {},
        },
        "runtime": {
            "device": "cpu",
            "seed": 23,
            "deterministic": True,
            "output_dir": str(output_dir),
        },
    }


def test_all_repository_yamls_pass_strict_validation():
    for path in Path("configs").glob("*.yaml"):
        load_experiment_config(path)


def test_prepare_experiment_seeds_model_and_data_before_creation(tmp_path):
    config = _synthetic_config(tmp_path)
    first = prepare_experiment(deepcopy(config))
    second = prepare_experiment(deepcopy(config))

    first_weight = next(first.model.parameters()).detach()
    second_weight = next(second.model.parameters()).detach()
    assert torch.equal(first_weight, second_weight)

    first_batch = next(iter(first.train_loader))
    second_batch = next(iter(second.train_loader))
    assert torch.equal(first_batch[0], second_batch[0])
    assert torch.equal(first_batch[1], second_batch[1])
