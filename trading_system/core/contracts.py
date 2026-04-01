from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    DRY_RUN = "dry-run"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Event:
    symbol: str
    timestamp: datetime
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    timestamp: datetime
    bars: Dict[str, Bar]
    history: Dict[str, List[Bar]] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price


@dataclass
class PortfolioState:
    cash: float
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    consecutive_errors: int = 0
    halted: bool = False


@dataclass
class Decision:
    bot_name: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    qty: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    symbol: str
    timestamp: datetime
    side: OrderSide
    qty: float
    price: float
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    bot_name: str
    mode: Mode
    decisions: List[Decision]
    fills: List[Fill]
    metrics: Dict[str, Any]
    halted: bool = False
    warnings: List[str] = field(default_factory=list)
