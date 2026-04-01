from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from trading_system.core.contracts import Decision, Fill
from trading_system.storage.attribution import AttributionRecord, AttributionStore

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
except ImportError:  # pragma: no cover
    TradingClient = None
    AlpacaOrderSide = None
    TimeInForce = None
    GetOrdersRequest = None
    MarketOrderRequest = None


@dataclass
class AlpacaConfig:
    key_env: str = "ALPACA_API_KEY"
    secret_env: str = "ALPACA_API_SECRET"
    paper: bool = True
    dry_run: bool = False
    attribution_db_path: str = "artifacts/attribution.sqlite3"


class DryRunExecutor:
    def __init__(self, attribution_store: Optional[AttributionStore] = None, run_id: str = "dry-run"):
        self.attribution_store = attribution_store
        self.run_id = run_id

    def execute(self, decisions: List[Decision], latest_prices: dict) -> List[Fill]:
        fills = []
        for decision in decisions:
            client_order_id = f"{decision.bot_name}-{uuid.uuid4().hex[:12]}"
            if self.attribution_store is not None:
                self.attribution_store.record(
                    AttributionRecord(
                        run_id=self.run_id,
                        bot_name=decision.bot_name,
                        symbol=decision.symbol,
                        broker="dry-run",
                        submitted_at=decision.timestamp,
                        client_order_id=client_order_id,
                        broker_order_id=client_order_id,
                        status="filled",
                        side=decision.side.value,
                        qty=decision.qty,
                        metadata=decision.metadata,
                    )
                )
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=latest_prices[decision.symbol],
                    metadata={**decision.metadata, "executor": "dry-run", "bot_name": decision.bot_name, "client_order_id": client_order_id},
                )
            )
        return fills


class AlpacaPaperExecutor:
    def __init__(self, config: AlpacaConfig, attribution_store: Optional[AttributionStore] = None, run_id: str = "paper"):
        self.config = config
        self.run_id = run_id
        self.attribution_store = attribution_store
        if TradingClient is None:
            raise RuntimeError("alpaca-py is not installed")
        key = os.environ.get(config.key_env)
        secret = os.environ.get(config.secret_env)
        if not key or not secret:
            raise RuntimeError("Missing Alpaca credentials in environment")
        self.client = TradingClient(key, secret, paper=config.paper)

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()

    def get_orders(self, status: str = "all"):
        if GetOrdersRequest is None:
            return []
        return self.client.get_orders(filter=GetOrdersRequest(status=status))

    def execute(self, decisions: List[Decision]) -> List[Fill]:
        fills = []
        for decision in decisions:
            client_order_id = f"{decision.bot_name}-{uuid.uuid4().hex[:12]}"
            order = MarketOrderRequest(
                symbol=decision.symbol,
                qty=decision.qty,
                side=AlpacaOrderSide(decision.side.value),
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
            placed = self.client.submit_order(order_data=order)
            if self.attribution_store is not None:
                self.attribution_store.record(
                    AttributionRecord(
                        run_id=self.run_id,
                        bot_name=decision.bot_name,
                        symbol=decision.symbol,
                        broker="alpaca",
                        submitted_at=decision.timestamp,
                        client_order_id=client_order_id,
                        broker_order_id=str(getattr(placed, "id", "")),
                        status=str(getattr(placed, "status", "submitted")),
                        side=decision.side.value,
                        qty=decision.qty,
                        metadata=decision.metadata,
                    )
                )
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=float(getattr(placed, "filled_avg_price", 0.0) or 0.0),
                    metadata={
                        **decision.metadata,
                        "broker": "alpaca",
                        "bot_name": decision.bot_name,
                        "client_order_id": client_order_id,
                        "alpaca_order_id": str(getattr(placed, "id", "")),
                        "alpaca_status": str(getattr(placed, "status", "submitted")),
                    },
                )
            )
        return fills
