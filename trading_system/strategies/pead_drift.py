from __future__ import annotations

from trading_system.core.bot import BaseStrategyBot
from trading_system.core.contracts import Decision, OrderSide


class PEADEventBot(BaseStrategyBot):
    strategy_name = "pead_drift"
    description = "Post-earnings announcement drift continuation bot"

    def evaluate(self, snapshot, portfolio):
        decisions = []
        for event in snapshot.events:
            if event.event_type != "earnings_surprise":
                continue
            surprise = float(event.payload.get("surprise", 0.0))
            reaction = float(event.payload.get("reaction", 0.0))
            if surprise > 0 and reaction > 0 and event.symbol in snapshot.bars:
                price = snapshot.bars[event.symbol].close
                budget = max(portfolio.cash * 0.1, 0)
                qty = budget / price if price else 0
                if qty > 0:
                    decisions.append(
                        Decision(
                            bot_name=self.context.name,
                            timestamp=snapshot.timestamp,
                            symbol=event.symbol,
                            side=OrderSide.BUY,
                            qty=qty,
                            rationale=f"Positive earnings surprise {surprise:.2f} with confirming reaction {reaction:.2f}",
                            metadata=event.payload,
                        )
                    )
        return decisions
