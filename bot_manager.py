from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from logger_setup import get_bot_logger
from trading_system.core.backtest import BacktestConfig, BacktestEngine
from trading_system.core.contracts import Event, PortfolioState
from trading_system.core.registry import create_bot, list_bots
from trading_system.core.risk import RiskLimits, RiskManager
from trading_system.data.alpaca_market_data import AlpacaHistoricalDataSource
from trading_system.data.events import PEADEventProvider
from trading_system.data.market_data import CsvHistoricalDataSource, build_market_snapshots
from trading_system.execution.alpaca import AlpacaConfig, AlpacaPaperExecutor, DryRunExecutor
from trading_system.leaderboard.snapshot import leaderboard_snapshot
from trading_system.reporting.artifacts import persist_run_artifacts
from trading_system.storage.attribution import AttributionStore

logger = get_bot_logger("bot_manager")
DEFAULT_DATA_DIR = Path("data/historical/daily")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_EVENT_PROVIDER = PEADEventProvider(
    fallback_events=[
        Event(symbol="NVDA", timestamp=datetime(2026, 1, 7), event_type="earnings_surprise", payload={"surprise": 0.15, "reaction": 0.04}),
    ]
)


class BotManager:
    def list_bots(self):
        return list_bots()

    def run_backtest(self, bot_name: str, data_dir: Path = DEFAULT_DATA_DIR, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        bot = create_bot(bot_name)
        risk = RiskManager(RiskLimits())
        run_id = f"{bot_name}-backtest-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        snapshots = self._load_snapshots(bot_name, bot.context.universe, data_dir)
        result = BacktestEngine(BacktestConfig()).run(bot, snapshots, risk)
        snapshot = leaderboard_snapshot([result], run_id=run_id, source="backtest")
        artifact_dir = persist_run_artifacts(artifacts_dir, run_id, result, snapshot, extra={"data_dir": str(data_dir)})
        logger.info("Backtest complete for %s ending equity %.2f artifacts=%s", bot_name, result.metrics["ending_equity"], artifact_dir)
        return result, snapshot, artifact_dir

    def run_dry(self, bot_name: str, data_dir: Path = DEFAULT_DATA_DIR, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        bot = create_bot(bot_name)
        run_id = f"{bot_name}-dry-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        snapshots = self._load_snapshots(bot_name, bot.context.universe, data_dir)
        latest = snapshots[-1]
        portfolio = PortfolioState(cash=100000.0, equity=100000.0)
        decisions = bot.evaluate(latest, portfolio)
        store = AttributionStore(artifacts_dir / "attribution.sqlite3")
        executor = DryRunExecutor(attribution_store=store, run_id=run_id)
        fills = executor.execute(decisions, {symbol: bar.close for symbol, bar in latest.bars.items()})
        payload = {"bot": bot_name, "run_id": run_id, "decisions": [decision.__dict__ for decision in decisions], "fills": [fill.__dict__ for fill in fills], "attribution": store.list_for_run(run_id)}
        (artifacts_dir / run_id).mkdir(parents=True, exist_ok=True)
        (artifacts_dir / run_id / "dry_run.json").write_text(json.dumps(payload, indent=2, default=str))
        return payload

    def run_paper(self, bot_name: str, data_dir: Path = DEFAULT_DATA_DIR, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        bot = create_bot(bot_name)
        run_id = f"{bot_name}-paper-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        snapshots = self._load_snapshots(bot_name, bot.context.universe, data_dir)
        latest = snapshots[-1]
        store = AttributionStore(artifacts_dir / "attribution.sqlite3")
        executor = AlpacaPaperExecutor(AlpacaConfig(), attribution_store=store, run_id=run_id)
        portfolio = executor.portfolio_state_from_reconciliation(bot_name=bot_name, universe=bot.context.universe, run_id=run_id)
        decisions = bot.evaluate(latest, portfolio)
        fills = executor.execute(decisions)
        reconciliation = executor.reconcile_state(bot_name=bot_name, universe=bot.context.universe, run_id=run_id)
        payload = {
            "bot": bot_name,
            "run_id": run_id,
            "decision_count": len(decisions),
            "fills": [fill.__dict__ for fill in fills],
            "reconciliation": reconciliation,
            "attribution": store.list_for_run(run_id),
        }
        (artifacts_dir / run_id).mkdir(parents=True, exist_ok=True)
        (artifacts_dir / run_id / "paper_run.json").write_text(json.dumps(payload, indent=2, default=str))
        return payload

    def sync_historical_data(self, symbols, data_dir: Path = DEFAULT_DATA_DIR, start: str = "2025-01-01", end: str = None):
        source = AlpacaHistoricalDataSource()
        return source.sync_bars(symbols=symbols, output_dir=data_dir, start=start, end=end)

    @staticmethod
    def _load_snapshots(bot_name: str, symbols, data_dir: Path):
        source = CsvHistoricalDataSource(data_dir)
        bars_by_symbol = source.load_bars(symbols)
        if not bars_by_symbol:
            raise RuntimeError(f"No historical bars loaded from {data_dir}")
        events = DEFAULT_EVENT_PROVIDER.get_events(bot_name=bot_name, symbols=symbols)
        return build_market_snapshots(bars_by_symbol, events=events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading prototype bot manager")
    parser.add_argument("--list", action="store_true", help="List official prototype bots")
    parser.add_argument("--backtest", type=str, help="Run backtest for one bot")
    parser.add_argument("--backtest-all", action="store_true", help="Run backtests for all official bots")
    parser.add_argument("--dry-run", type=str, help="Generate decisions/fills without broker execution")
    parser.add_argument("--paper", type=str, help="Submit current bot decisions to Alpaca paper trading")
    parser.add_argument("--sync-historical", nargs="+", help="Sync historical bars for the provided symbols from Alpaca into --data-dir")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Historical sync start date/time (YYYY-MM-DD or ISO-8601)")
    parser.add_argument("--end", type=str, help="Historical sync end date/time (YYYY-MM-DD or ISO-8601)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing historical CSV bars")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR, help="Directory for run artifacts")
    args = parser.parse_args()

    manager = BotManager()
    if args.list:
        for name in manager.list_bots():
            print(name)
    elif args.backtest:
        _, snapshot, artifact_dir = manager.run_backtest(args.backtest, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
        print(json.dumps({"artifact_dir": str(artifact_dir), "snapshot": snapshot}, indent=2))
    elif args.backtest_all:
        outputs = []
        for name in manager.list_bots():
            _, snapshot, artifact_dir = manager.run_backtest(name, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
            outputs.append({"bot": name, "artifact_dir": str(artifact_dir), "snapshot": snapshot})
        print(json.dumps(outputs, indent=2))
    elif args.dry_run:
        print(json.dumps(manager.run_dry(args.dry_run, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir), indent=2, default=str))
    elif args.paper:
        print(json.dumps(manager.run_paper(args.paper, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir), indent=2, default=str))
    elif args.sync_historical:
        print(json.dumps(manager.sync_historical_data(args.sync_historical, data_dir=args.data_dir, start=args.start, end=args.end), indent=2))
    else:
        parser.print_help()
