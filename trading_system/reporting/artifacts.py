from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from trading_system.core.contracts import RunResult


def persist_run_artifacts(base_dir: Path | str, run_id: str, result: RunResult, snapshot: Dict[str, Any], extra: Dict[str, Any] | None = None) -> Path:
    artifact_dir = Path(base_dir) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "metrics.json": result.metrics,
        "decisions.json": [_json_ready(item) for item in result.decisions],
        "trade_log.json": [_json_ready(item) for item in result.fills],
        "leaderboard_snapshot.json": snapshot,
        "run_manifest.json": {
            "run_id": run_id,
            "bot_name": result.bot_name,
            "mode": result.mode.value,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "halted": result.halted,
            "warnings": result.warnings,
            "extra": extra or {},
        },
    }

    for filename, payload in payloads.items():
        (artifact_dir / filename).write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))

    return artifact_dir


def _json_ready(value: Any):
    if is_dataclass(value):
        return {key: _json_ready(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
