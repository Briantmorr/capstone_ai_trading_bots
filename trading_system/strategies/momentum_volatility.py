from __future__ import annotations

from statistics import mean, pstdev

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Decision, OrderSide


class MomentumVolatilityBot(BaseStrategyBot):
    strategy_name = "momentum_volatility"
    description = "Cross-sectional momentum with inverse-volatility sizing"

    def evaluate(self, snapshot, portfolio):
        lookback = self.context.params.get("history", {})
        history = lookback if isinstance(lookback, dict) else {}
        scores = []
        for symbol in self.context.universe:
            series = history.get(symbol, [])
            if len(series) < 3 or symbol not in snapshot.bars:
                continue
            momentum = (series[-1] / series[0]) - 1
            returns = [(series[i] / series[i - 1]) - 1 for i in range(1, len(series))]
            volatility = pstdev(returns) if len(returns) > 1 else 0.01
            if volatility <= 0:
                volatility = 0.01
            scores.append((symbol, momentum / volatility, volatility))

        if not scores:
            return []

        scores.sort(key=lambda item: item[1], reverse=True)
        winners = scores[: min(2, len(scores))]
        budget = max(portfolio.cash, 0) / max(len(winners), 1)
        decisions = []
        for symbol, score, volatility in winners:
            price = snapshot.bars[symbol].close
            qty = budget / price
            decisions.append(
                Decision(
                    bot_name=self.context.name,
                    timestamp=snapshot.timestamp,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    qty=qty,
                    rationale=f"Top momentum rank with vol-adjusted score {score:.3f}",
                    metadata={"score": score, "volatility": volatility},
                )
            )
        return decisions
