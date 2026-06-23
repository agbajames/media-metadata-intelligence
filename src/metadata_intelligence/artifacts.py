"""Model and report artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> None:
    """Persist a fitted model with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str | Path) -> Any:
    """Load a persisted model with joblib."""

    return joblib.load(Path(path))


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON data with stable indentation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
