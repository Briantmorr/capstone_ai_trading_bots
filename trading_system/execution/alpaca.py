from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from trading_system.core.contracts import Decision, Fill, OrderSide

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except ImportError:  # pragma: no cover
    TradingClient = None
    TimeInForce = None
    MarketOrderRequest = None


@dataclass
class AlpacaConfig:
    key_env: str = "ALPACA_API_KEY"
    secret_env: str = "ALPACA_API_SECRET"
    paper: bool = True
    dry_run: bool = False


class DryRunExecutor:
    def execute(self, decisions: List[Decision], latest_prices: dict) -> List[Fill]:
        fills = []
        for decision in decisions:
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=latest_prices[decision.symbol],
                    metadata={**decision.metadata, "executor": "dry-run", "bot_name": decision.bot_name},
                )
            )
        return fills


class AlpacaPaperExecutor:
    def __init__(self, config: AlpacaConfig):
        self.config = config
        if TradingClient is None:
            raise RuntimeError("alpaca-py is not installed")
        key = os.environ.get(config.key_env)
        secret = os.environ.get(config.secret_env)
        if not key or not secret:
            raise RuntimeError("Missing Alpaca credentials in environment")
        self.client = TradingClient(key, secret, paper=config.paper)

    def execute(self, decisions: List[Decision]) -> List[Fill]:
        fills = []
        for decision in decisions:
            order = MarketOrderRequest(
                symbol=decision.symbol,
                qty=decision.qty,
                side=decision.side.value,
                time_in_force=TimeInForce.DAY,
            )
            placed = self.client.submit_order(order_data=order)
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=0.0,
                    metadata={
                        **decision.metadata,
                        "broker": "alpaca",
                        "bot_name": decision.bot_name,
                        "alpaca_order_id": str(getattr(placed, 'id', '')),
                    },
                )
            )
        return fills
