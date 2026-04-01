from __future__ import annotations

from statistics import mean

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Decision, OrderSide


class IntradayMeanReversionBot(BaseStrategyBot):
    strategy_name = "intraday_mean_reversion"
    description = "Hourly mean reversion with trend and volatility filter"

    def evaluate(self, snapshot, portfolio):
        history = self.context.params.get("history", {})
        decisions = []
        for symbol in self.context.universe:
            series = history.get(symbol, [])
            if len(series) < 5 or symbol not in snapshot.bars:
                continue
            anchor = mean(series[-5:])
            current = snapshot.bars[symbol].close
            trend = (series[-1] / series[0]) - 1
            deviation = (current / anchor) - 1 if anchor else 0
            if deviation < -0.015 and abs(trend) < 0.03:
                qty = max((portfolio.cash * 0.1) / current, 0)
                if qty > 0:
                    decisions.append(
                        Decision(
                            bot_name=self.context.name,
                            timestamp=snapshot.timestamp,
                            symbol=symbol,
                            side=OrderSide.BUY,
                            qty=qty,
                            rationale=f"Price deviated {deviation:.2%} below short-term mean with tame trend {trend:.2%}",
                            metadata={"anchor": anchor, "deviation": deviation, "trend": trend},
                        )
                    )
        return decisions
