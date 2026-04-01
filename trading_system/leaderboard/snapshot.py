from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional

from trading_system.core.contracts import RunResult

SNAPSHOT_CONTRACT_VERSION = "2026-04-01"


def leaderboard_row(result: RunResult) -> Dict:
    return {
        "bot_name": result.bot_name,
        "mode": result.mode.value,
        "ending_equity": result.metrics.get("ending_equity"),
        "ending_cash": result.metrics.get("ending_cash"),
        "trade_count": result.metrics.get("trade_count", 0),
        "halted": result.halted,
        "warnings": result.warnings,
        "last_equity_point": (result.metrics.get("equity_curve") or [None])[-1],
    }


def leaderboard_snapshot(results: Iterable[RunResult], run_id: Optional[str] = None, source: str = "backtest", account: Optional[Dict] = None) -> Dict:
    rows = [leaderboard_row(result) for result in results]
    rows.sort(key=lambda row: row.get("ending_equity") or 0, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "contract_version": SNAPSHOT_CONTRACT_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "source": source,
        "account": account or {},
        "bots": rows,
    }
