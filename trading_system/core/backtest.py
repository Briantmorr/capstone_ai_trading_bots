from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Decision, Fill, MarketSnapshot, Mode, OrderSide, PortfolioState, Position, RunResult
from trading_system.core.risk import RiskManager


@dataclass
class BacktestConfig:
    initial_cash: float = 100000.0
    fee_per_order: float = 0.0
    slippage_bps: float = 5.0


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, bot: BaseStrategyBot, snapshots: Iterable[MarketSnapshot], risk: RiskManager) -> RunResult:
        ordered = list(snapshots)
        portfolio = PortfolioState(cash=self.config.initial_cash, equity=self.config.initial_cash)
        decisions: List[Decision] = []
        fills: List[Fill] = []
        equity_curve = []
        pending: List[Decision] = []

        for snapshot in ordered:
            self._execute_pending(snapshot, pending, portfolio, fills)
            pending = []
            self._mark_to_market(portfolio, snapshot)
            bot_decisions = bot.evaluate(snapshot, portfolio)
            prices = {symbol: bar.close for symbol, bar in snapshot.bars.items()}
            approved = risk.filter_decisions(bot_decisions, portfolio, prices)
            decisions.extend(approved)
            pending.extend(approved)
            self._mark_to_market(portfolio, snapshot)
            equity_curve.append({"timestamp": snapshot.timestamp.isoformat(), "equity": portfolio.equity})

        metrics = {
            "ending_cash": portfolio.cash,
            "ending_equity": portfolio.equity,
            "trade_count": len(fills),
            "equity_curve": equity_curve,
            "pending_decision_count": len(pending),
        }
        warnings = []
        if pending:
            warnings.append("Final snapshot produced decisions that were not filled because no later bar was available.")
        return RunResult(bot_name=bot.context.name, mode=Mode.BACKTEST, decisions=decisions, fills=fills, metrics=metrics, halted=portfolio.halted, warnings=warnings)

    def _execute_pending(self, snapshot: MarketSnapshot, pending: List[Decision], portfolio: PortfolioState, fills: List[Fill]) -> None:
        for decision in pending:
            bar = snapshot.bars.get(decision.symbol)
            if bar is None:
                continue
            fill = self._fill(decision, bar.open, snapshot.timestamp)
            self._apply_fill(portfolio, fill)
            fills.append(fill)

    def _fill(self, decision: Decision, reference_price: float, fill_timestamp) -> Fill:
        slip_multiplier = 1 + (self.config.slippage_bps / 10000.0)
        price = reference_price * slip_multiplier if decision.side == OrderSide.BUY else reference_price / slip_multiplier
        return Fill(
            symbol=decision.symbol,
            timestamp=fill_timestamp,
            side=decision.side,
            qty=decision.qty,
            price=price,
            fees=self.config.fee_per_order,
            metadata=decision.metadata,
        )

    @staticmethod
    def _apply_fill(portfolio: PortfolioState, fill: Fill) -> None:
        signed_qty = fill.qty if fill.side == OrderSide.BUY else -fill.qty
        if fill.side == OrderSide.BUY:
            portfolio.cash -= fill.qty * fill.price + fill.fees
        else:
            portfolio.cash += (fill.qty * fill.price) - fill.fees

        existing = portfolio.positions.get(fill.symbol)
        if existing:
            new_qty = existing.qty + signed_qty
            if abs(new_qty) < 1e-9:
                portfolio.positions.pop(fill.symbol, None)
            else:
                if signed_qty > 0:
                    total_cost = (existing.qty * existing.avg_price) + (fill.qty * fill.price)
                    existing.avg_price = total_cost / new_qty
                existing.qty = new_qty
                existing.market_price = fill.price
        elif signed_qty > 0:
            portfolio.positions[fill.symbol] = Position(symbol=fill.symbol, qty=signed_qty, avg_price=fill.price, market_price=fill.price)

    @staticmethod
    def _mark_to_market(portfolio: PortfolioState, snapshot: MarketSnapshot) -> None:
        equity = portfolio.cash
        for symbol, position in list(portfolio.positions.items()):
            bar = snapshot.bars.get(symbol)
            if bar:
                position.market_price = bar.close
            equity += position.market_value
        portfolio.equity = equity
