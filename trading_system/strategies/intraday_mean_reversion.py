from __future__ import annotations

from statistics import mean, pstdev

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Decision, OrderSide


class IntradayMeanReversionBot(BaseStrategyBot):
    strategy_name = "intraday_mean_reversion"
    description = "Hourly mean reversion with trend and volatility filter"

    def evaluate(self, snapshot, portfolio):
        decisions = []
        for symbol in self.context.universe:
            history = snapshot.history.get(symbol, [])
            series = [bar.close for bar in history]
            if len(series) < 5 or symbol not in snapshot.bars:
                continue
            anchor = mean(series[-5:])
            current = snapshot.bars[symbol].close
            trend = (series[-1] / series[0]) - 1 if series[0] else 0.0
            returns = [(series[i] / series[i - 1]) - 1 for i in range(1, len(series)) if series[i - 1]]
            realized_vol = pstdev(returns[-5:]) if len(returns) > 1 else 0.0
            deviation = (current / anchor) - 1 if anchor else 0.0
            if deviation < -0.015 and abs(trend) < 0.03 and realized_vol < 0.03:
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
                            metadata={"anchor": anchor, "deviation": deviation, "trend": trend, "realized_vol": realized_vol},
                        )
                    )
        return decisions
