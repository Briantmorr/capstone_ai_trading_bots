from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AttributionRecord:
    run_id: str
    bot_name: str
    symbol: str
    broker: str
    submitted_at: datetime
    client_order_id: str
    broker_order_id: str = ""
    status: str = "submitted"
    side: str = ""
    qty: float = 0.0
    metadata: Dict | None = None


class AttributionStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_attribution (
                    client_order_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    bot_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    broker_order_id TEXT,
                    submitted_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    side TEXT,
                    qty REAL,
                    metadata_json TEXT
                )
                """
            )

    def record(self, record: AttributionRecord) -> None:
        import json

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO order_attribution (
                    client_order_id, run_id, bot_name, symbol, broker, broker_order_id,
                    submitted_at, status, side, qty, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.client_order_id,
                    record.run_id,
                    record.bot_name,
                    record.symbol,
                    record.broker,
                    record.broker_order_id,
                    record.submitted_at.isoformat(),
                    record.status,
                    record.side,
                    record.qty,
                    json.dumps(record.metadata or {}, sort_keys=True),
                ),
            )

    def list_for_run(self, run_id: str) -> List[Dict]:
        import json

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT client_order_id, run_id, bot_name, symbol, broker, broker_order_id, submitted_at, status, side, qty, metadata_json FROM order_attribution WHERE run_id = ? ORDER BY submitted_at ASC",
                (run_id,),
            ).fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "client_order_id": row[0],
                    "run_id": row[1],
                    "bot_name": row[2],
                    "symbol": row[3],
                    "broker": row[4],
                    "broker_order_id": row[5],
                    "submitted_at": row[6],
                    "status": row[7],
                    "side": row[8],
                    "qty": row[9],
                    "metadata": json.loads(row[10] or "{}"),
                }
            )
        return results
