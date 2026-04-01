import unittest
from datetime import datetime
from pathlib import Path

from trading_system.core.backtest import BacktestConfig, BacktestEngine
from trading_system.core.bot import BaseStrategyBot, BotContext
from trading_system.core.contracts import Bar, Decision, Event, MarketSnapshot, PortfolioState, RunResult, Mode, OrderSide
from trading_system.core.risk import RiskLimits, RiskManager
from trading_system.data.market_data import CsvHistoricalDataSource, build_market_snapshots
from trading_system.leaderboard.snapshot import leaderboard_snapshot
from trading_system.reporting.artifacts import persist_run_artifacts
from trading_system.storage.attribution import AttributionRecord, AttributionStore
from trading_system.strategies.intraday_mean_reversion import IntradayMeanReversionBot
from trading_system.strategies.momentum_volatility import MomentumVolatilityBot
from trading_system.strategies.pead_drift import PEADEventBot

FIXTURES = Path(__file__).parent / "fixtures" / "historical" / "daily"


class HistoryInspectBot(BaseStrategyBot):
    def __init__(self):
        super().__init__(BotContext(name="history_inspector", cadence="daily", universe=["AAPL"], params={}))
        self.seen_history_lengths = []

    def evaluate(self, snapshot, portfolio):
        history = snapshot.history.get("AAPL", [])
        self.seen_history_lengths.append(len(history))
        if len(history) >= 2:
            return [
                Decision(
                    bot_name=self.context.name,
                    timestamp=snapshot.timestamp,
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    qty=1,
                    rationale="inspect next-bar fill",
                )
            ]
        return []


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
            history={"NVDA": [Bar("NVDA", datetime(2026, 1, 1), 98, 100, 97, 99, 900_000)]},
            events=[Event(symbol="NVDA", timestamp=datetime(2026, 1, 2), event_type="earnings_surprise", payload={"surprise": 0.2, "reaction": 0.03})],
        )
        decisions = bot.evaluate(snapshot, PortfolioState(cash=100000, equity=100000))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].symbol, "NVDA")

    def test_intraday_mean_reversion_needs_dislocation(self):
        bot = IntradayMeanReversionBot(BotContext(name="intraday_mean_reversion", cadence="hourly", universe=["AAPL"], params={}))
        history = [
            Bar("AAPL", datetime(2026, 1, 1, hour), 100, 100, 99, close, 1_000_000)
            for hour, close in enumerate([100, 100, 100, 99.8, 99.5], start=9)
        ]
        snapshot = MarketSnapshot(
            timestamp=datetime(2026, 1, 3),
            bars={"AAPL": Bar("AAPL", datetime(2026, 1, 3), 98, 99, 97, 97.8, 1_000_000)},
            history={"AAPL": history},
        )
        decisions = bot.evaluate(snapshot, PortfolioState(cash=100000, equity=100000))
        self.assertEqual(len(decisions), 1)

    def test_csv_ingestion_builds_causal_snapshots(self):
        bars_by_symbol = CsvHistoricalDataSource(FIXTURES).load_bars(["AAPL"])
        snapshots = build_market_snapshots(bars_by_symbol, lookback_bars=10)
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(len(snapshots[0].history["AAPL"]), 0)
        self.assertEqual(len(snapshots[1].history["AAPL"]), 1)
        self.assertEqual(snapshots[1].history["AAPL"][-1].close, 101.0)
        self.assertEqual(snapshots[2].bars["AAPL"].close, 105.0)

    def test_backtest_uses_next_bar_open_fill_and_no_future_history(self):
        bars_by_symbol = CsvHistoricalDataSource(FIXTURES).load_bars(["AAPL"])
        snapshots = build_market_snapshots(bars_by_symbol, lookback_bars=10)
        bot = HistoryInspectBot()
        result = BacktestEngine(BacktestConfig(slippage_bps=0.0)).run(bot, snapshots, RiskManager(RiskLimits(max_symbol_weight=1.0)))
        self.assertEqual(bot.seen_history_lengths, [0, 1, 2, 3])
        self.assertEqual(len(result.fills), 1)
        self.assertEqual(result.fills[0].timestamp, snapshots[3].timestamp)
        self.assertEqual(result.fills[0].price, snapshots[3].bars["AAPL"].open)

    def test_backtest_runs(self):
        bars_by_symbol = CsvHistoricalDataSource(Path("data/historical/daily")).load_bars(["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
        snapshots = build_market_snapshots(bars_by_symbol)
        bot = MomentumVolatilityBot(BotContext(name="momentum_volatility", cadence="daily", universe=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"], params={}))
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(bot, snapshots, RiskManager(RiskLimits()))
        self.assertEqual(result.mode, Mode.BACKTEST)
        self.assertIn("ending_equity", result.metrics)

    def test_artifact_generation_persists_expected_files(self):
        with self.subTest("artifact persistence"):
            import tempfile
            from pathlib import Path as _Path

            with tempfile.TemporaryDirectory() as tmp:
                base = _Path(tmp)
                result = RunResult(bot_name="a", mode=Mode.BACKTEST, decisions=[], fills=[], metrics={"ending_equity": 110_000, "trade_count": 2})
                snapshot = leaderboard_snapshot([result], run_id="run-123")
                artifact_dir = persist_run_artifacts(base, "run-123", result, snapshot)
                self.assertTrue((artifact_dir / "metrics.json").exists())
                self.assertTrue((artifact_dir / "decisions.json").exists())
                self.assertTrue((artifact_dir / "trade_log.json").exists())
                self.assertTrue((artifact_dir / "leaderboard_snapshot.json").exists())
                self.assertTrue((artifact_dir / "run_manifest.json").exists())

    def test_attribution_store_records_and_reads_orders(self):
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            store = AttributionStore(_Path(tmp) / "attribution.sqlite3")
            store.record(
                AttributionRecord(
                    run_id="run-1",
                    bot_name="momentum_volatility",
                    symbol="AAPL",
                    broker="alpaca",
                    submitted_at=datetime(2026, 1, 2, 9, 30),
                    client_order_id="client-123",
                    broker_order_id="broker-456",
                    status="submitted",
                    side="buy",
                    qty=10,
                    metadata={"reason": "ranked"},
                )
            )
            rows = store.list_for_run("run-1")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["client_order_id"], "client-123")
            self.assertEqual(rows[0]["broker_order_id"], "broker-456")

    def test_leaderboard_snapshot_ranks_by_equity(self):
        snapshot = leaderboard_snapshot([
            RunResult(bot_name="b", mode=Mode.BACKTEST, decisions=[], fills=[], metrics={"ending_equity": 90_000, "trade_count": 1}),
            RunResult(bot_name="a", mode=Mode.BACKTEST, decisions=[], fills=[], metrics={"ending_equity": 110_000, "trade_count": 2}),
        ], run_id="run-xyz")
        self.assertEqual(snapshot["bots"][0]["bot_name"], "a")
        self.assertEqual(snapshot["bots"][0]["rank"], 1)
        self.assertEqual(snapshot["contract_version"], "2026-04-01")


if __name__ == "__main__":
    unittest.main()
