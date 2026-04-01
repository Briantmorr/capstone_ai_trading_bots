from __future__ import annotations

from typing import Dict

from trading_system.core.contracts import RunResult


def leaderboard_row(result: RunResult) -> Dict:
    return {
        "bot_name": result.bot_name,
        "mode": result.mode.value,
        "ending_equity": result.metrics.get("ending_equity"),
        "trade_count": result.metrics.get("trade_count", 0),
        "halted": result.halted,
        "warnings": result.warnings,
    }


def leaderboard_snapshot(results):
    rows = [leaderboard_row(result) for result in results]
    rows.sort(key=lambda row: row.get("ending_equity") or 0, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"bots": rows}
