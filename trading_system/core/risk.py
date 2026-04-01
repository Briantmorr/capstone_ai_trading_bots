from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from trading_system.core.contracts import Decision, OrderSide, PortfolioState


@dataclass
class RiskLimits:
    allow_short: bool = False
    max_symbol_weight: float = 0.20
    max_gross_exposure: float = 1.00
    daily_loss_cap: float = 0.03
    max_consecutive_errors: int = 3


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def filter_decisions(self, decisions: Iterable[Decision], portfolio: PortfolioState, latest_prices: dict) -> List[Decision]:
        approved: List[Decision] = []
        if portfolio.halted or portfolio.consecutive_errors >= self.limits.max_consecutive_errors:
            portfolio.halted = True
            return approved

        if portfolio.equity > 0 and portfolio.daily_pnl / portfolio.equity <= -self.limits.daily_loss_cap:
            portfolio.halted = True
            return approved

        gross_exposure = self._gross_exposure(portfolio)
        for decision in decisions:
            if decision.side == OrderSide.SELL and not self.limits.allow_short:
                position = portfolio.positions.get(decision.symbol)
                if position is None or position.qty <= 0:
                    continue
                decision.qty = min(decision.qty, position.qty)

            price = latest_prices.get(decision.symbol)
            if not price or portfolio.equity <= 0:
                continue

            notional = decision.qty * price
            if notional / portfolio.equity > self.limits.max_symbol_weight:
                decision.qty = max((portfolio.equity * self.limits.max_symbol_weight) / price, 0)

            if decision.qty <= 0:
                continue

            projected_gross = gross_exposure + (decision.qty * price / portfolio.equity)
            if projected_gross > self.limits.max_gross_exposure:
                continue

            approved.append(decision)
        return approved

    @staticmethod
    def _gross_exposure(portfolio: PortfolioState) -> float:
        if portfolio.equity <= 0:
            return 0.0
        return sum(abs(position.market_value) for position in portfolio.positions.values()) / portfolio.equity
