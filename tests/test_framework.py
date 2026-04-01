import unittest
from datetime import datetime

from trading_system.core.backtest import BacktestConfig, BacktestEngine
from trading_system.core.bot import BotContext
from trading_system.core.contracts import Bar, Decision, Event, MarketSnapshot, OrderSide, PortfolioState, RunResult, Mode
from trading_system.core.risk import RiskLimits, RiskManager
from trading_system.leaderboard.snapshot import leaderboard_snapshot
from trading_system.strategies.intraday_mean_reversion import IntradayMeanReversionBot
from trading_system.strategies.momentum_volatility import MomentumVolatilityBot
from trading_system.strategies.pead_drift import PEADEventBot


class FrameworkTests(unittest.TestCase):
    def test_risk_caps_single_symbol_weight(self):
        timestamp = datetime(2026, 1, 1)
        decision = Decision(bot_name="test", timestamp=timestamp, symbol="AAPL", side=OrderSide.BUY, qty=1000, rationale="oversized")
        portfolio = PortfolioState(cash=100000, equity=100000)
        manager = RiskManager(RiskLimits(max_symbol_weight=0.2))
        approved = manager.filter_decisions([decision], portfolio, {"AAPL": 100})
        self.assertEqual(len(approved), 1)
        self.assertEqual(round(approved[0].qty, 2), 200.0)

    def test_pead_only_trades_on_event(self):
        bot = PEADEventBot(BotContext(name="pead_drift", cadence="event", universe=["NVDA"], params={}))
        snapshot = MarketSnapshot(
            timestamp=datetime(2026, 1, 2),
            bars={"NVDA": Bar("NVDA", datetime(2026, 1, 2), 100, 101, 99, 100, 1_000_000)},
            events=[Event(symbol="NVDA", timestamp=datetime(2026, 1, 2), event_type="earnings_surprise", payload={"surprise": 0.2, "reaction": 0.03})],
        )
        decisions = bot.evaluate(snapshot, PortfolioState(cash=100000, equity=100000))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].symbol, "NVDA")

    def test_intraday_mean_reversion_needs_dislocation(self):
        bot = IntradayMeanReversionBot(BotContext(name="intraday_mean_reversion", cadence="hourly", universe=["AAPL"], params={"history": {"AAPL": [100, 100, 100, 99.8, 99.5, 97.8]}}))
        snapshot = MarketSnapshot(
            timestamp=datetime(2026, 1, 3),
            bars={"AAPL": Bar("AAPL", datetime(2026, 1, 3), 98, 99, 97, 97.8, 1_000_000)},
        )
        decisions = bot.evaluate(snapshot, PortfolioState(cash=100000, equity=100000))
        self.assertEqual(len(decisions), 1)

    def test_backtest_runs(self):
        bot = MomentumVolatilityBot(BotContext(name="momentum_volatility", cadence="daily", universe=["AAPL", "MSFT"], params={"history": {"AAPL": [100, 103, 105, 108], "MSFT": [100, 100, 101, 101.5]}}))
        snapshots = [
            MarketSnapshot(timestamp=datetime(2026, 1, 1), bars={"AAPL": Bar("AAPL", datetime(2026, 1, 1), 100, 101, 99, 105, 1_000_000), "MSFT": Bar("MSFT", datetime(2026, 1, 1), 100, 101, 99, 101.5, 1_000_000)}),
            MarketSnapshot(timestamp=datetime(2026, 1, 2), bars={"AAPL": Bar("AAPL", datetime(2026, 1, 2), 105, 106, 104, 108, 1_000_000), "MSFT": Bar("MSFT", datetime(2026, 1, 2), 101, 102, 100, 101.5, 1_000_000)}),
        ]
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(bot, snapshots, RiskManager(RiskLimits()))
        self.assertEqual(result.mode, Mode.BACKTEST)
        self.assertIn("ending_equity", result.metrics)

    def test_leaderboard_snapshot_ranks_by_equity(self):
        snapshot = leaderboard_snapshot([
            RunResult(bot_name="b", mode=Mode.BACKTEST, decisions=[], fills=[], metrics={"ending_equity": 90_000, "trade_count": 1}),
            RunResult(bot_name="a", mode=Mode.BACKTEST, decisions=[], fills=[], metrics={"ending_equity": 110_000, "trade_count": 2}),
        ])
        self.assertEqual(snapshot["bots"][0]["bot_name"], "a")
        self.assertEqual(snapshot["bots"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
