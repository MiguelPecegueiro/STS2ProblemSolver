"""Train behavioral cloning policy on data/decisions.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from training.actions import vocab_metadata
from training.dataset import build_datasets
from training.features import FEATURE_DIM, feature_layout
from training.model import PolicyNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECISIONS = PROJECT_ROOT / "data" / "decisions.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "policy_net.pt"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "model_config.json"


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def _weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_sample = nn.functional.cross_entropy(logits, targets, reduction="none")
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate(
    model: PolicyNet,
    X: np.ndarray,
    y: np.ndarray,
    state_types: list[str],
) -> dict[str, float | dict[str, float]]:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = len(y)
    by_state: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    batch = 512
    for start in range(0, total, batch):
        end = min(start + batch, total)
        xb = torch.from_numpy(X[start:end]).to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        for i, pred in enumerate(preds):
            global_i = start + i
            truth = int(y[global_i])
            st = state_types[global_i]
            if int(pred) == truth:
                correct += 1
                by_state[st][0] += 1
            by_state[st][1] += 1

    overall = correct / total if total else 0.0
    per_state = {
        st: (hits / n if n else 0.0) for st, (hits, n) in sorted(by_state.items())
    }
    return {"accuracy": overall, "per_state_type": per_state, "num_samples": total}


def train(
    *,
    decisions_path: Path,
    model_path: Path,
    config_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    min_run_score_percentile: float,
    min_game_version: str | None,
    human_weight: float,
    val_fraction: float,
    seed: int,
    device_name: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, val_ds, vocab, meta = build_datasets(
        decisions_path,
        min_run_score_percentile=min_run_score_percentile,
        min_game_version=min_game_version,
        human_weight=human_weight,
        val_fraction=val_fraction,
        seed=seed,
    )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = PolicyNet(
        input_dim=FEATURE_DIM,
        num_actions=len(vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _make_loader(
        train_ds.X, train_ds.y, train_ds.w, batch_size=batch_size, shuffle=True
    )
    val_loader = _make_loader(
        val_ds.X, val_ds.y, val_ds.w, batch_size=batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for xb, yb, wb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = _weighted_loss(logits, yb, wb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * float(wb.sum().item())
            train_weight_sum += float(wb.sum().item())
            pbar.set_postfix(loss=f"{train_loss_sum / max(train_weight_sum, 1e-6):.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                logits = model(xb)
                loss = _weighted_loss(logits, yb, wb)
                val_loss_sum += float(loss.item()) * float(wb.sum().item())
                val_weight_sum += float(wb.sum().item())

        train_loss = train_loss_sum / max(train_weight_sum, 1e-6)
        val_loss = val_loss_sum / max(val_weight_sum, 1e-6)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": FEATURE_DIM,
                    "num_actions": len(vocab),
                },
                model_path,
            )

    # Load best weights for eval
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_metrics = evaluate(model, train_ds.X, train_ds.y, train_ds.state_types)
    val_metrics = evaluate(model, val_ds.X, val_ds.y, val_ds.state_types)

    config = {
        "feature_layout": feature_layout(),
        "action_vocab": vocab_metadata(vocab),
        "training": {
            "decisions_path": str(decisions_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "min_run_score_percentile": min_run_score_percentile,
            "min_game_version": min_game_version,
            "human_weight": human_weight,
            "val_fraction": val_fraction,
            "seed": seed,
            "dataset_meta": meta,
            "history": history,
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== Evaluation ===")
    print(f"Train accuracy: {train_metrics['accuracy']:.1%} ({train_metrics['num_samples']} samples)")
    print("Train by state_type:")
    for st, acc in train_metrics["per_state_type"].items():
        print(f"  {st}: {acc:.1%}")
    print(f"Val accuracy:   {val_metrics['accuracy']:.1%} ({val_metrics['num_samples']} samples)")
    print("Val by state_type:")
    for st, acc in val_metrics["per_state_type"].items():
        print(f"  {st}: {acc:.1%}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved config: {config_path}")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC policy on decisions.jsonl")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-out", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--min-run-score-percentile",
        type=float,
        default=25.0,
        help="Drop runs below this run_score percentile (0-100)",
    )
    parser.add_argument(
        "--min-game-version",
        default=None,
        metavar="ID",
        help="Only train on runs/decisions with game_version >= this (YYYY.MM.DD)",
    )
    parser.add_argument(
        "--human-weight",
        type=float,
        default=3.0,
        help="Sample-weight multiplier for runs with source=human vs agent (default 3.0)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train(
        decisions_path=args.decisions,
        model_path=args.model_out,
        config_path=args.config_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_run_score_percentile=args.min_run_score_percentile,
        min_game_version=args.min_game_version,
        human_weight=args.human_weight,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
