from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Fill, MarketSnapshot, Mode, OrderSide, PortfolioState, Position, RunResult
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
        portfolio = PortfolioState(cash=self.config.initial_cash, equity=self.config.initial_cash)
        decisions = []
        fills: List[Fill] = []
        equity_curve = []

        for snapshot in snapshots:
            self._mark_to_market(portfolio, snapshot)
            bot_decisions = bot.evaluate(snapshot, portfolio)
            prices = {symbol: bar.close for symbol, bar in snapshot.bars.items()}
            approved = risk.filter_decisions(bot_decisions, portfolio, prices)
            decisions.extend(approved)
            for decision in approved:
                fill = self._fill(decision, prices[decision.symbol])
                self._apply_fill(portfolio, fill)
                fills.append(fill)
            self._mark_to_market(portfolio, snapshot)
            equity_curve.append({"timestamp": snapshot.timestamp.isoformat(), "equity": portfolio.equity})

        metrics = {
            "ending_cash": portfolio.cash,
            "ending_equity": portfolio.equity,
            "trade_count": len(fills),
            "equity_curve": equity_curve,
        }
        return RunResult(bot_name=bot.context.name, mode=Mode.BACKTEST, decisions=decisions, fills=fills, metrics=metrics, halted=portfolio.halted)

    def _fill(self, decision, last_price: float) -> Fill:
        slip_multiplier = 1 + (self.config.slippage_bps / 10000.0)
        price = last_price * slip_multiplier if decision.side == OrderSide.BUY else last_price / slip_multiplier
        return Fill(
            symbol=decision.symbol,
            timestamp=decision.timestamp,
            side=decision.side,
            qty=decision.qty,
            price=price,
            fees=self.config.fee_per_order,
            metadata=decision.metadata,
        )

    @staticmethod
    def _apply_fill(portfolio: PortfolioState, fill: Fill) -> None:
        signed_qty = fill.qty if fill.side == OrderSide.BUY else -fill.qty
        cash_delta = fill.qty * fill.price + fill.fees
        if fill.side == OrderSide.BUY:
            portfolio.cash -= cash_delta
        else:
            portfolio.cash += (fill.qty * fill.price) - fill.fees

        existing = portfolio.positions.get(fill.symbol)
        if existing:
            new_qty = existing.qty + signed_qty
            if abs(new_qty) < 1e-9:
                portfolio.positions.pop(fill.symbol, None)
            else:
                existing.avg_price = fill.price if signed_qty > 0 else existing.avg_price
                existing.qty = new_qty
                existing.market_price = fill.price
        else:
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
