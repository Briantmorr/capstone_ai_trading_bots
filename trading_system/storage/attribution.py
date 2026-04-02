from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


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
        return self._query(
            "SELECT client_order_id, run_id, bot_name, symbol, broker, broker_order_id, submitted_at, status, side, qty, metadata_json FROM order_attribution WHERE run_id = ? ORDER BY submitted_at ASC",
            (run_id,),
        )

    def list_for_bot(self, bot_name: str) -> List[Dict]:
        return self._query(
            "SELECT client_order_id, run_id, bot_name, symbol, broker, broker_order_id, submitted_at, status, side, qty, metadata_json FROM order_attribution WHERE bot_name = ? ORDER BY submitted_at ASC",
            (bot_name,),
        )

    def _query(self, query: str, params: tuple) -> List[Dict]:
        import json

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
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


def estimate_bot_pnl_allocation(account: Dict, positions: List[Dict], orders: List[Dict], attribution: List[Dict], bot_name: str) -> Dict:
    """
    Estimate bot-scoped exposure and PnL inside a shared broker account.

    Allocation rules are intentionally simple and reviewable:
    - only filled orders attributable to the target bot participate
    - net filled quantity is derived from attributable buy/sell orders per symbol
    - live position quantity, market value, and unrealized PnL are allocated by the
      bot's net-quantity share against the current live position size
    - account cash is allocated pro-rata by attributable market value share

    This is a prototype approximation for leaderboard/reporting use; it is not a
    tax-lot accounting engine.
    """

    attributed_orders = [row for row in attribution if row.get("bot_name") == bot_name]
    if not attributed_orders:
        return {
            "bot_name": bot_name,
            "allocation_method": "filled-order-net-qty-share",
            "allocated_cash": 0.0,
            "allocated_market_value": 0.0,
            "allocated_unrealized_pl": 0.0,
            "estimated_realized_pnl": 0.0,
            "allocated_equity": 0.0,
            "symbols": [],
        }

    orders_by_client_id = {row.get("client_order_id"): row for row in orders}
    position_by_symbol = {row.get("symbol"): row for row in positions}
    per_symbol_filled_qty: Dict[str, float] = {}
    per_symbol_buy_notional: Dict[str, float] = {}
    per_symbol_sell_notional: Dict[str, float] = {}

    for row in attributed_orders:
        order = orders_by_client_id.get(row.get("client_order_id"), {})
        symbol = row.get("symbol") or order.get("symbol")
        status = str(order.get("status") or row.get("status") or "")
        if status != "filled" or not symbol:
            continue
        filled_qty = float(order.get("filled_qty", row.get("qty", 0.0)) or 0.0)
        fill_price = float(order.get("filled_avg_price", 0.0) or 0.0)
        side = str(order.get("side") or row.get("side") or "buy")
        signed_qty = filled_qty if side == "buy" else -filled_qty
        per_symbol_filled_qty[symbol] = per_symbol_filled_qty.get(symbol, 0.0) + signed_qty
        notional = filled_qty * fill_price
        if side == "buy":
            per_symbol_buy_notional[symbol] = per_symbol_buy_notional.get(symbol, 0.0) + notional
        else:
            per_symbol_sell_notional[symbol] = per_symbol_sell_notional.get(symbol, 0.0) + notional

    symbol_rows = []
    allocated_market_value = 0.0
    allocated_unrealized_pl = 0.0
    estimated_realized_pnl = 0.0

    for symbol, net_qty in sorted(per_symbol_filled_qty.items()):
        live = position_by_symbol.get(symbol)
        current_qty = abs(float(live.get("qty", 0.0) or 0.0)) if live else 0.0
        if current_qty <= 0:
            allocation_ratio = 0.0
            allocated_qty = 0.0
            market_value = 0.0
            unrealized = 0.0
            realized = max(per_symbol_sell_notional.get(symbol, 0.0) - per_symbol_buy_notional.get(symbol, 0.0), 0.0)
        else:
            ratio_basis = max(current_qty, abs(net_qty))
            allocation_ratio = min(abs(net_qty) / ratio_basis, 1.0) if ratio_basis else 0.0
            allocated_qty = current_qty * allocation_ratio
            market_value = float(live.get("market_value", 0.0) or 0.0) * allocation_ratio
            unrealized = float(live.get("unrealized_pl", 0.0) or 0.0) * allocation_ratio
            realized = 0.0
        allocated_market_value += market_value
        allocated_unrealized_pl += unrealized
        estimated_realized_pnl += realized
        symbol_rows.append(
            {
                "symbol": symbol,
                "net_filled_qty": net_qty,
                "allocated_qty": allocated_qty,
                "allocation_ratio": allocation_ratio,
                "allocated_market_value": market_value,
                "allocated_unrealized_pl": unrealized,
                "estimated_realized_pnl": realized,
            }
        )

    total_gross_market_value = sum(abs(float(row.get("market_value", 0.0) or 0.0)) for row in positions)
    exposure_share = (allocated_market_value / max(total_gross_market_value, 1e-9)) if total_gross_market_value else 0.0
    allocated_cash = float(account.get("cash", 0.0) or 0.0) * exposure_share
    allocated_equity = allocated_cash + allocated_market_value + allocated_unrealized_pl + estimated_realized_pnl

    return {
        "bot_name": bot_name,
        "allocation_method": "filled-order-net-qty-share",
        "allocated_cash": allocated_cash,
        "allocated_market_value": allocated_market_value,
        "allocated_unrealized_pl": allocated_unrealized_pl,
        "estimated_realized_pnl": estimated_realized_pnl,
        "allocated_equity": allocated_equity,
        "symbols": symbol_rows,
    }
