"""YAML-first experiment API shared by CLI scripts and notebooks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.Model_A_OutGridNet import MaxOutNet
from src.Model_B_OutGridNet import OutlookerFrontGridNet
from src.model.downsampling import DownsampleConfig
from src.stage_config import StageCfg
from src.training.autocast import seed_everything
from src.training.chekpoints import load_checkpoint
from src.training.one_epoch_train import evaluate_one_epoch
from src.training.train_full_model import train_model


TOP_LEVEL_KEYS = {"model", "data", "training", "runtime"}
MODEL_KEYS = {
    "type", "name", "pretrained", "num_classes", "in_ch", "stem_dim",
    "dpr_max", "stages", "downsample", "outlooker_front_depth",
}
DATA_KEYS = {
    "dataset", "data_dir", "hf_name", "batch_size", "num_workers",
    "val_split", "pin_memory", "img_size", "drop_last", "augmentation",
    "num_samples",
}
AUGMENTATION_KEYS = {
    "random_crop", "horizontal_flip", "randaugment", "ra_num_ops",
    "ra_magnitude", "random_erasing", "random_erasing_p",
}
TRAINING_KEYS = {
    "epochs", "optimizer", "optimizer_betas", "optimizer_eps",
    "optimizer_momentum", "optimizer_nesterov", "scheduler", "lr",
    "weight_decay", "autocast_dtype", "use_amp",
    "grad_clip_norm", "warmup_ratio", "min_lr", "label_smoothing",
    "print_every", "save_path", "last_path", "resume_path", "mixup_alpha",
    "cutmix_alpha", "mix_prob", "channels_last", "early_stop",
    "early_stop_metric", "early_stop_patience", "early_stop_min_delta",
    "early_stop_require_monotonic",
}
RUNTIME_KEYS = {
    "device", "seed", "deterministic", "output_dir", "evaluate_test",
    "load_best_for_test",
}


def _reject_unknown(section: str, values: dict, allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {', '.join(unknown)}")


def validate_experiment_config(config: dict) -> None:
    _reject_unknown("root", config, TOP_LEVEL_KEYS)
    for required in ("model", "data", "training", "runtime"):
        if required not in config or not isinstance(config[required], dict):
            raise ValueError(f"Config section '{required}' is required and must be a mapping")

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    runtime_cfg = config["runtime"]
    _reject_unknown("model", model_cfg, MODEL_KEYS)
    _reject_unknown("data", data_cfg, DATA_KEYS)
    _reject_unknown("training", train_cfg, TRAINING_KEYS)
    _reject_unknown("runtime", runtime_cfg, RUNTIME_KEYS)

    aug_cfg = data_cfg.get("augmentation", {})
    if not isinstance(aug_cfg, dict):
        raise ValueError("data.augmentation must be a mapping")
    _reject_unknown("data.augmentation", aug_cfg, AUGMENTATION_KEYS)

    model_type = str(model_cfg.get("type", "model_a")).lower()
    if model_type in ("timm", "baseline", "timm_baseline"):
        if not model_cfg.get("name"):
            raise ValueError("model.name is required when model.type is 'timm'")
    elif not model_cfg.get("stages"):
        raise ValueError("model.stages must contain at least one stage")

    val_split = float(data_cfg.get("val_split", 0.0))
    if not 0.0 <= val_split < 1.0:
        raise ValueError("data.val_split must be in [0, 1)")
    for key in ("mix_prob", "label_smoothing", "warmup_ratio"):
        value = float(train_cfg.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"training.{key} must be in [0, 1]")
    if int(runtime_cfg.get("seed", 7)) < 0:
        raise ValueError("runtime.seed must be non-negative")
    if str(train_cfg.get("optimizer", "adamw")).lower() not in {
        "adamw", "adam", "sgd"
    }:
        raise ValueError("training.optimizer must be adamw, adam, or sgd")
    if str(train_cfg.get("scheduler", "warmup_cosine")).lower() not in {
        "warmup_cosine", "constant", "none"
    }:
        raise ValueError(
            "training.scheduler must be warmup_cosine, constant, or none"
        )
    betas = train_cfg.get("optimizer_betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("training.optimizer_betas must contain exactly two values")


def load_experiment_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    validate_experiment_config(config)
    return config


def build_stages(stage_cfgs: list[dict]) -> list[StageCfg]:
    return [StageCfg(**cfg) for cfg in stage_cfgs]


def build_model(model_cfg: dict, img_size: int) -> torch.nn.Module:
    model_type = str(model_cfg.get("type", "model_a")).lower()
    num_classes = int(model_cfg.get("num_classes", 100))
    if model_type in ("timm", "baseline", "timm_baseline"):
        from src.timm_baselines import build_timm_baseline

        return build_timm_baseline(
            name=str(model_cfg["name"]),
            num_classes=num_classes,
            img_size=img_size,
            pretrained=bool(model_cfg.get("pretrained", False)),
        )

    stages = build_stages(model_cfg.get("stages", []))
    down_cfg = DownsampleConfig(**model_cfg.get("downsample", {}))
    common = dict(
        num_classes=num_classes,
        stages=stages,
        in_ch=int(model_cfg.get("in_ch", 3)),
        stem_dim=int(model_cfg.get("stem_dim", 64)),
        dpr_max=float(model_cfg.get("dpr_max", 0.1)),
        down_cfg=down_cfg,
    )
    if model_type in ("a", "model_a", "maxout", "outgrid"):
        return MaxOutNet(**common)
    if model_type in ("b", "model_b", "outlooker_front", "front"):
        return OutlookerFrontGridNet(
            outlooker_front_depth=int(model_cfg.get("outlooker_front_depth", 2)),
            **common,
        )
    raise ValueError(f"Unknown model.type '{model_type}'")


def build_dataloaders(data_cfg: dict, num_classes: int, seed: int):
    dataset = str(data_cfg.get("dataset", "cifar100")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    img_size = int(data_cfg.get("img_size", 32))
    aug = data_cfg.get("augmentation", {})
    augmentation_args = dict(
        random_crop=bool(aug.get("random_crop", True)),
        horizontal_flip=bool(
            aug.get("horizontal_flip", dataset != "svhn")
        ),
        randaugment=bool(aug.get("randaugment", True)),
        ra_num_ops=int(aug.get("ra_num_ops", 2)),
        ra_magnitude=int(aug.get("ra_magnitude", 7)),
        random_erasing=bool(aug.get("random_erasing", True)),
        random_erasing_p=float(aug.get("random_erasing_p", 0.25)),
    )
    common = dict(
        batch_size=batch_size,
        data_dir=str(data_cfg.get("data_dir", "./data")),
        num_workers=num_workers,
        val_split=float(data_cfg.get("val_split", 0.0)),
        pin_memory=pin_memory,
        img_size=img_size,
        drop_last=bool(data_cfg.get("drop_last", False)),
        seed=int(seed),
        **augmentation_args,
    )
    if dataset == "cifar100":
        from src.data.load_cifrar100 import get_cifar100_dataloaders

        return get_cifar100_dataloaders(**common)
    if dataset == "svhn":
        from src.data.load_svhn import get_svhn_dataloaders

        return get_svhn_dataloaders(**common)
    if dataset in ("tinyimagenet200", "tinyimagenet", "tiny-imagenet"):
        from src.data.load_tinyimagenet import get_tinyimagenet200_hf_dataloaders

        return get_tinyimagenet200_hf_dataloaders(
            hf_name=str(data_cfg.get("hf_name", "zh-plus/tiny-imagenet")),
            **common,
        )
    if dataset == "synthetic":
        generator = torch.Generator().manual_seed(seed)
        count = int(data_cfg.get("num_samples", 256))
        images = torch.randn(count, 3, img_size, img_size, generator=generator)
        labels = torch.randint(
            0, num_classes, (count,), generator=generator
        )
        dataset_obj = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        return loader, None, None
    raise ValueError(
        "data.dataset must be cifar100, svhn, tinyimagenet200, or synthetic"
    )


def _resolve_device(requested: str) -> str:
    device = requested.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return device


def _output_path(output_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


@dataclass
class PreparedExperiment:
    config: dict
    model: torch.nn.Module
    train_loader: Any
    val_loader: Any
    test_loader: Any
    device: str
    output_dir: Path
    best_path: Path
    last_path: Path


def prepare_experiment(config_or_path: dict | str | Path) -> PreparedExperiment:
    config = (
        deepcopy(config_or_path)
        if isinstance(config_or_path, dict)
        else load_experiment_config(config_or_path)
    )
    validate_experiment_config(config)
    runtime_cfg = config["runtime"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    seed = int(runtime_cfg.get("seed", 7))
    seed_everything(seed, deterministic=bool(runtime_cfg.get("deterministic", True)))

    device = _resolve_device(str(runtime_cfg.get("device", "cuda")))
    output_dir = Path(runtime_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_cfg, img_size=int(data_cfg.get("img_size", 32)))
    loaders = build_dataloaders(
        data_cfg, num_classes=int(model_cfg.get("num_classes", 100)), seed=seed
    )
    best_path = _output_path(
        output_dir, str(train_cfg.get("save_path", "best_model.pt"))
    )
    last_path = _output_path(
        output_dir, str(train_cfg.get("last_path", "last_model.pt"))
    )
    return PreparedExperiment(
        config=config,
        model=model,
        train_loader=loaders[0],
        val_loader=loaders[1],
        test_loader=loaders[2],
        device=device,
        output_dir=output_dir,
        best_path=best_path,
        last_path=last_path,
    )


def run_experiment(config_or_path: dict | str | Path) -> dict:
    """Train from YAML and optionally evaluate the selected best checkpoint."""
    prepared = prepare_experiment(config_or_path)
    train_cfg = prepared.config["training"]
    runtime_cfg = prepared.config["runtime"]
    num_classes = int(prepared.config["model"].get("num_classes", 100))
    use_amp = bool(train_cfg.get("use_amp", True)) and prepared.device == "cuda"

    history, model = train_model(
        model=prepared.model,
        train_loader=prepared.train_loader,
        epochs=int(train_cfg.get("epochs", 1)),
        val_loader=prepared.val_loader,
        device=prepared.device,
        optimizer_name=str(train_cfg.get("optimizer", "adamw")),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        optimizer_betas=tuple(
            float(value)
            for value in train_cfg.get("optimizer_betas", [0.9, 0.999])
        ),
        optimizer_eps=float(train_cfg.get("optimizer_eps", 1e-8)),
        optimizer_momentum=float(train_cfg.get("optimizer_momentum", 0.9)),
        optimizer_nesterov=bool(train_cfg.get("optimizer_nesterov", False)),
        scheduler_name=str(train_cfg.get("scheduler", "warmup_cosine")),
        autocast_dtype=str(train_cfg.get("autocast_dtype", "fp16")),
        use_amp=use_amp,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        min_lr=float(train_cfg.get("min_lr", 0.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        print_every=int(train_cfg.get("print_every", 100)),
        save_path=str(prepared.best_path),
        last_path=str(prepared.last_path),
        resume_path=train_cfg.get("resume_path"),
        mixup_alpha=float(train_cfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(train_cfg.get("cutmix_alpha", 0.0)),
        mix_prob=float(train_cfg.get("mix_prob", 0.0)),
        num_classes=num_classes,
        channels_last=bool(train_cfg.get("channels_last", False)),
        early_stop=bool(train_cfg.get("early_stop", True)),
        early_stop_metric=str(train_cfg.get("early_stop_metric", "top1")),
        early_stop_patience=int(train_cfg.get("early_stop_patience", 10)),
        early_stop_min_delta=float(train_cfg.get("early_stop_min_delta", 0.0)),
        early_stop_require_monotonic=bool(
            train_cfg.get("early_stop_require_monotonic", False)
        ),
    )

    test_metrics = None
    test_loss = None
    if bool(runtime_cfg.get("evaluate_test", True)) and prepared.test_loader is not None:
        if (
            bool(runtime_cfg.get("load_best_for_test", True))
            and prepared.best_path.exists()
        ):
            load_checkpoint(
                str(prepared.best_path), model, map_location=prepared.device
            )
        test_loss, test_metrics = evaluate_one_epoch(
            model=model,
            dataloader=prepared.test_loader,
            device=prepared.device,
            autocast_dtype=str(train_cfg.get("autocast_dtype", "fp16")),
            use_amp=use_amp,
            label_smoothing=0.0,
            channels_last=bool(train_cfg.get("channels_last", False)),
        )
        print(f"[Test] loss {test_loss:.4f} | metrics {test_metrics}")

    return {
        "config": prepared.config,
        "model": model,
        "history": history,
        "train_loader": prepared.train_loader,
        "val_loader": prepared.val_loader,
        "test_loader": prepared.test_loader,
        "best_path": prepared.best_path,
        "last_path": prepared.last_path,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
