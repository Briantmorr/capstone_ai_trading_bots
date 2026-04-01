from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from trading_system.core.contracts import Decision, MarketSnapshot, PortfolioState


@dataclass
class BotContext:
    name: str
    cadence: str
    universe: List[str]
    params: Dict[str, float] = field(default_factory=dict)


class BaseStrategyBot:
    strategy_name = "base"
    description = "Base strategy contract"

    def __init__(self, context: BotContext):
        self.context = context

    def evaluate(self, snapshot: MarketSnapshot, portfolio: PortfolioState) -> List[Decision]:
        raise NotImplementedError

    def spec(self) -> Dict[str, str]:
        return {
            "name": self.context.name,
            "strategy_name": self.strategy_name,
            "description": self.description,
            "cadence": self.context.cadence,
        }
