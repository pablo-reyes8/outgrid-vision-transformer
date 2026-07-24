import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_cifrar100 import get_cifar100_dataloaders
from src.timm_baselines import build_timm_baseline, list_timm_baselines
from src.training.autocast import seed_everything
from src.training.train_full_model import train_model


def _slugify(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-100 baselines (CIFAR-32 setup)")
    parser.add_argument(
        "--models",
        default=",".join(list_timm_baselines()),
        help="Comma-separated list of baseline models",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="outputs_baselines")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)

    parser.add_argument("--autocast-dtype", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--print-every", type=int, default=400)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = str(args.device).lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No models provided in --models")

    for name in model_names:
        print(f"\n=== Baseline: {name} ===")
        seed_everything(args.seed, deterministic=True)

        train_loader, val_loader, _ = get_cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
            val_split=args.val_split,
            img_size=args.img_size,
            seed=args.seed,
        )

        model = build_timm_baseline(
            name, num_classes=100, img_size=args.img_size, device=device)
        n_params = _count_params(model)
        print(f"Trainable parameters: {n_params:,}")

        tag = _slugify(name)
        save_path = output_dir / f"best_{tag}.pt"
        last_path = output_dir / f"last_{tag}.pt"

        autocast_dtype = args.autocast_dtype
        if autocast_dtype == "auto":
            autocast_dtype = "fp16" if device == "cuda" else "fp32"

        print(
            "Train config: "
            f"epochs={args.epochs}, lr={args.lr}, weight_decay={args.weight_decay}, "
            f"autocast_dtype={autocast_dtype}, use_amp={(not args.no_amp) and (device == 'cuda')}, "
            f"grad_clip_norm={args.grad_clip_norm}, warmup_ratio={args.warmup_ratio}, "
            f"min_lr={args.min_lr}, label_smoothing={args.label_smoothing}, "
            f"mix_prob={args.mix_prob}, mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, "
            f"channels_last={not args.no_channels_last}"
        )

        train_model(
            model=model,
            train_loader=train_loader,
            epochs=args.epochs,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            autocast_dtype=autocast_dtype,
            use_amp=(not args.no_amp) and (device == "cuda"),
            grad_clip_norm=args.grad_clip_norm,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
            label_smoothing=args.label_smoothing,
            print_every=args.print_every,
            save_path=str(save_path),
            last_path=str(last_path),
            resume_path=None,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
            num_classes=100,
            channels_last=not args.no_channels_last,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
