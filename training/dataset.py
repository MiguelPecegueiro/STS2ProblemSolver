"""Load decisions.jsonl, filter by run quality, apply score-based sample weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.actions import action_to_key, build_action_vocab, encode_action
from training.features import encode_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECISIONS_PATH = PROJECT_ROOT / "data" / "decisions.jsonl"
DEFAULT_RUNS_PATH = PROJECT_ROOT / "data" / "runs.jsonl"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, dict):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_runs_index(runs_path: Path) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    if not runs_path.exists():
        return scores, sources
    with runs_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("run_id")
            if not rid:
                continue
            rid = str(rid)
            score = row.get("run_score")
            if score is not None:
                scores[rid] = float(score)
            source = row.get("source")
            if source is not None:
                sources[rid] = str(source)
    return scores, sources


def load_run_scores(runs_path: Path) -> dict[str, float]:
    scores, _ = _load_runs_index(runs_path)
    return scores


def load_run_sources(runs_path: Path) -> dict[str, str]:
    _, sources = _load_runs_index(runs_path)
    return sources


def _run_score_for_row(row: dict, run_scores: dict[str, float]) -> float | None:
    outcome = row.get("run_outcome") or {}
    if outcome.get("run_score") is not None:
        return float(outcome["run_score"])
    rid = row.get("run_id")
    if rid and str(rid) in run_scores:
        return run_scores[str(rid)]
    return None


def load_decision_rows(
    decisions_path: Path,
    *,
    runs_path: Path | None = None,
    min_run_score: float | None = None,
    min_run_score_percentile: float = 25.0,
) -> tuple[list[dict], dict[str, float]]:
    """Load rows with valid actions; return (rows, run_id -> score)."""
    run_scores = load_run_scores(runs_path or DEFAULT_RUNS_PATH)

    # Fill run scores from decision outcomes
    for line in decisions_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = row.get("run_id")
        if not rid:
            continue
        score = _run_score_for_row(row, run_scores)
        if score is not None:
            run_scores[str(rid)] = max(run_scores.get(str(rid), score), score)

    if min_run_score is None and run_scores:
        values = sorted(run_scores.values())
        if values:
            pct = float(np.clip(min_run_score_percentile, 0.0, 100.0))
            idx = int((pct / 100.0) * (len(values) - 1))
            min_run_score = values[idx]

    allowed_runs: set[str] = set()
    if min_run_score is not None:
        allowed_runs = {rid for rid, sc in run_scores.items() if sc >= min_run_score}

    rows: list[dict] = []
    with decisions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("run_id") or "")
            if not rid:
                continue
            if allowed_runs and rid not in allowed_runs:
                continue
            if not row.get("action_taken"):
                continue
            if not action_to_key(row.get("action_taken"), snapshot=row.get("state_snapshot")):
                continue
            rows.append(row)

    return rows, run_scores


def _run_source_for_row(row: dict, run_sources: dict[str, str]) -> str:
    rid = str(row.get("run_id") or "")
    if rid and rid in run_sources:
        return str(run_sources[rid])
    return ""


def compute_sample_weights(
    rows: list[dict],
    run_scores: dict[str, float],
    run_sources: dict[str, str] | None = None,
    *,
    human_weight: float = 3.0,
) -> np.ndarray:
    """Higher weight for decisions from higher-scoring runs; boost human runs."""
    weights = np.ones(len(rows), dtype=np.float32)
    sources = run_sources or {}

    if run_scores:
        scores = []
        for row in rows:
            sc = _run_score_for_row(row, run_scores)
            scores.append(sc if sc is not None else 0.0)
        arr = np.array(scores, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            norm = (arr - lo) / (hi - lo)
            weights = 0.1 + 0.9 * norm

    if human_weight != 1.0:
        for i, row in enumerate(rows):
            if _run_source_for_row(row, sources).lower() == "human":
                weights[i] *= float(human_weight)

    return weights.astype(np.float32)


class DecisionDataset:
    """In-memory dataset of (features, label, weight)."""

    def __init__(
        self,
        rows: list[dict],
        vocab: dict[str, int],
        weights: np.ndarray | None = None,
        *,
        unknown_action_id: int | None = None,
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.unknown_action_id = unknown_action_id

        xs: list[np.ndarray] = []
        ys: list[int] = []
        ws: list[float] = []
        run_ids: list[str] = []
        state_types: list[str] = []

        for i, row in enumerate(rows):
            label = encode_action(
                row.get("action_taken"),
                vocab,
                snapshot=row.get("state_snapshot"),
                unknown_index=unknown_action_id,
            )
            if label is None:
                continue
            xs.append(
                encode_snapshot(
                    row.get("state_snapshot"),
                    state_type=str(row.get("state_type") or ""),
                    floor=int(row.get("floor") or 0),
                    act=int(row.get("act") or 1),
                    immediate_reward=_safe_float(row.get("immediate_reward")),
                )
            )
            ys.append(label)
            ws.append(float(weights[i]) if weights is not None else 1.0)
            run_ids.append(str(row.get("run_id") or ""))
            state_types.append(str(row.get("state_type") or "unknown"))

        self.X = np.stack(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
        self.y = np.array(ys, dtype=np.int64)
        self.w = np.array(ws, dtype=np.float32)
        self.run_ids = run_ids
        self.state_types = state_types

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, float]:
        return self.X[idx], int(self.y[idx]), float(self.w[idx])


def train_val_split_by_run(
    dataset: DecisionDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/val index arrays; no run appears in both splits."""
    rng = np.random.default_rng(seed)
    runs = sorted(set(dataset.run_ids))
    rng.shuffle(runs)
    n_val = max(1, int(len(runs) * val_fraction))
    val_runs = set(runs[:n_val])
    train_idx = np.array(
        [i for i, rid in enumerate(dataset.run_ids) if rid not in val_runs],
        dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, rid in enumerate(dataset.run_ids) if rid in val_runs],
        dtype=np.int64,
    )
    return train_idx, val_idx


def build_datasets(
    decisions_path: Path = DEFAULT_DECISIONS_PATH,
    *,
    runs_path: Path | None = None,
    min_run_score: float | None = None,
    min_run_score_percentile: float = 25.0,
    human_weight: float = 3.0,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DecisionDataset, DecisionDataset, dict[str, int], dict[str, Any]]:
    rows, run_scores = load_decision_rows(
        decisions_path,
        runs_path=runs_path,
        min_run_score=min_run_score,
        min_run_score_percentile=min_run_score_percentile,
    )
    if not rows:
        raise ValueError(f"No training rows after filtering: {decisions_path}")

    run_sources = load_run_sources(runs_path or DEFAULT_RUNS_PATH)
    vocab = build_action_vocab(decisions_path)
    weights = compute_sample_weights(
        rows,
        run_scores,
        run_sources,
        human_weight=human_weight,
    )
    full = DecisionDataset(rows, vocab, weights)

    train_idx, val_idx = train_val_split_by_run(full, val_fraction=val_fraction, seed=seed)

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_weights = weights[train_idx] if len(train_idx) else weights[:0]
    val_weights = weights[val_idx] if len(val_idx) else weights[:0]

    train_ds = DecisionDataset(train_rows, vocab, train_weights)
    val_ds = DecisionDataset(val_rows, vocab, val_weights)

    human_rows = sum(
        1 for row in rows if _run_source_for_row(row, run_sources).lower() == "human"
    )
    meta = {
        "num_rows_total": len(rows),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_runs": len(run_scores),
        "num_human_rows": human_rows,
        "human_weight": human_weight,
        "min_run_score_used": min_run_score,
        "min_run_score_percentile": min_run_score_percentile,
        "run_score_min": float(min(run_scores.values())) if run_scores else 0.0,
        "run_score_max": float(max(run_scores.values())) if run_scores else 0.0,
    }
    return train_ds, val_ds, vocab, meta
