"""Frozen feature scaling for PG-TL SFCC inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model import ALL_FEATURES


def default_scaler_path() -> Path:
    """Return the scaler stats path for a source checkout."""

    return Path(__file__).resolve().parents[1] / "models" / "scaler_stats.json"


def load_scaler_stats(path: str | Path | None = None) -> dict:
    """Load frozen StandardScaler statistics."""

    stats_path = default_scaler_path() if path is None else Path(path)
    with stats_path.open("r", encoding="utf-8") as fh:
        stats = json.load(fh)
    if list(stats["feature_order"]) != ALL_FEATURES:
        raise ValueError(
            "Scaler feature order does not match the model runtime: "
            f"{stats['feature_order']} != {ALL_FEATURES}"
        )
    return stats


class FrozenStandardScaler:
    """Minimal StandardScaler-compatible transformer backed by JSON stats."""

    def __init__(self, stats: Mapping):
        self.feature_order = list(stats["feature_order"])
        self.mean = np.asarray(stats["mean"], dtype=np.float64)
        self.scale = np.asarray(stats["scale"], dtype=np.float64)
        if self.mean.shape != self.scale.shape:
            raise ValueError("Scaler mean and scale arrays must have the same shape.")
        if len(self.feature_order) != self.mean.size:
            raise ValueError("Scaler feature_order length must match mean/scale length.")

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "FrozenStandardScaler":
        return cls(load_scaler_stats(path))

    def transform_records(self, records: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Transform raw soil-property dictionaries into scaled model features."""

        rows = []
        for idx, record in enumerate(records):
            row = []
            for name in self.feature_order:
                if name not in record:
                    raise KeyError(f"Record {idx} is missing required feature '{name}'.")
                value = float(record[name])
                if name == "SOM":
                    value = math.log1p(value * 100.0)
                elif name == "S_a":
                    value = math.log1p(value)
                row.append(value)
            rows.append(row)
        raw = np.asarray(rows, dtype=np.float64)
        return (raw - self.mean) / self.scale
