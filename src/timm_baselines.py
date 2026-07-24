"""Canonical timm baselines adapted for small-resolution image datasets."""

from collections.abc import Callable

import torch
import torch.nn as nn
import timm


def _deit_tiny_patch4(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=4,
    )


def _deit_small_patch4(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=4,
    )


def _swin_tiny_patch2(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        window_size=4,
    )
    out_channels = model.patch_embed.proj.out_channels
    model.patch_embed.proj = nn.Conv2d(
        3, out_channels, kernel_size=2, stride=2, padding=0
    )
    model.patch_embed.patch_size = (2, 2)
    return model


def _convnext_tiny_cifar(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    del img_size
    model = timm.create_model(
        "convnext_tiny", pretrained=pretrained, num_classes=num_classes
    )
    model.stem[0] = nn.Conv2d(
        3, 96, kernel_size=2, stride=2, padding=0
    )
    return model


def _efficientnetv2_s_cifar(
    num_classes: int, img_size: int, pretrained: bool
) -> nn.Module:
    del img_size
    model = timm.create_model(
        "efficientnetv2_s", pretrained=pretrained, num_classes=num_classes
    )
    model.conv_stem = nn.Conv2d(
        3,
        model.conv_stem.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return model


def _maxvit_tiny_cifar(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    model = timm.create_model(
        "maxvit_tiny_tf_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )
    conv1, conv2 = model.stem.conv1, model.stem.conv2
    model.stem.conv1 = nn.Conv2d(
        conv1.in_channels,
        conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.stem.conv2 = nn.Conv2d(
        conv2.in_channels,
        conv2.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return model


def _maxvit_nano_cifar(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    model = timm.create_model(
        "maxvit_tiny_tf_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        embed_dim=[64, 96, 192, 384],
    )
    model.stem.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.stem.norm1 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.1)
    model.stem.conv2 = nn.Conv2d(
        64, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    return model


def _resnet_cifar(
    timm_name: str, num_classes: int, img_size: int, pretrained: bool
) -> nn.Module:
    del img_size
    model = timm.create_model(
        timm_name, pretrained=pretrained, num_classes=num_classes
    )
    model.conv1 = nn.Conv2d(
        3,
        model.conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def _resnet18_cifar(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    return _resnet_cifar("resnet18", num_classes, img_size, pretrained)


def _resnet50_cifar(num_classes: int, img_size: int, pretrained: bool) -> nn.Module:
    return _resnet_cifar("resnet50", num_classes, img_size, pretrained)


BASELINE_BUILDERS: dict[str, Callable[[int, int, bool], nn.Module]] = {
    "deit_tiny_patch4": _deit_tiny_patch4,
    "deit_small_patch4": _deit_small_patch4,
    "swin_tiny_patch2": _swin_tiny_patch2,
    "convnext_tiny_cifar": _convnext_tiny_cifar,
    "efficientnetv2_s_cifar": _efficientnetv2_s_cifar,
    "maxvit_nano_cifar": _maxvit_nano_cifar,
    "maxvit_tiny_cifar": _maxvit_tiny_cifar,
    "resnet18_cifar": _resnet18_cifar,
    "resnet50_cifar": _resnet50_cifar,
}

ALIASES = {
    "deit_tiny": "deit_tiny_patch4",
    "deit_small": "deit_small_patch4",
    "swin_tiny": "swin_tiny_patch2",
    "convnext_tiny": "convnext_tiny_cifar",
    "efficientnetv2_s": "efficientnetv2_s_cifar",
    "maxvit_nano": "maxvit_nano_cifar",
    "maxvit_tiny": "maxvit_tiny_cifar",
    "resnet18": "resnet18_cifar",
    "resnet50": "resnet50_cifar",
}


def list_timm_baselines() -> tuple[str, ...]:
    return tuple(BASELINE_BUILDERS)


def build_timm_baseline(
    name: str,
    num_classes: int,
    img_size: int,
    pretrained: bool = False,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Build one named, small-image baseline with a consistent stem."""
    key = ALIASES.get(name.lower(), name.lower())
    if key not in BASELINE_BUILDERS:
        available = ", ".join(list_timm_baselines())
        raise ValueError(f"Unknown timm baseline '{name}'. Available: {available}")
    model = BASELINE_BUILDERS[key](int(num_classes), int(img_size), bool(pretrained))
    return model.to(device) if device is not None else model

