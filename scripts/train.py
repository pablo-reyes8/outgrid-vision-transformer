import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import load_experiment_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YAML experiment")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", choices=["a", "b", "model_a", "model_b"])
    parser.add_argument("--baseline", help="Canonical name from src.timm_baselines")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--val-split", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    runtime_cfg = cfg["runtime"]

    if args.model:
        model_cfg["type"] = args.model
    if args.baseline:
        model_cfg["type"] = "timm"
        model_cfg["name"] = args.baseline
        model_cfg.pop("stages", None)
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.data_dir is not None:
        data_cfg["data_dir"] = args.data_dir
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.img_size is not None:
        data_cfg["img_size"] = args.img_size
    if args.val_split is not None:
        data_cfg["val_split"] = args.val_split
    if args.device is not None:
        runtime_cfg["device"] = args.device
    if args.output_dir is not None:
        runtime_cfg["output_dir"] = args.output_dir
    if args.resume is not None:
        train_cfg["resume_path"] = args.resume
    if args.seed is not None:
        runtime_cfg["seed"] = args.seed
    if args.no_amp:
        train_cfg["use_amp"] = False
    if args.non_deterministic:
        runtime_cfg["deterministic"] = False
    if args.no_test:
        runtime_cfg["evaluate_test"] = False

    result = run_experiment(cfg)
    print("Training complete. History keys:", sorted(result["history"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
