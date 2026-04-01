from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from logger_setup import get_bot_logger
from trading_system.core.backtest import BacktestConfig, BacktestEngine
from trading_system.core.contracts import Bar, Event, MarketSnapshot, Mode
from trading_system.core.registry import create_bot, list_bots
from trading_system.core.risk import RiskLimits, RiskManager
from trading_system.execution.alpaca import DryRunExecutor
from trading_system.leaderboard.snapshot import leaderboard_snapshot

logger = get_bot_logger("bot_manager")


class BotManager:
    def list_bots(self):
        return list_bots()

    def run_backtest(self, bot_name: str):
        bot = create_bot(bot_name, params={"history": self._demo_history(bot_name)})
        engine = BacktestEngine(BacktestConfig())
        risk = RiskManager(RiskLimits())
        result = engine.run(bot, self._demo_snapshots(bot_name), risk)
        logger.info("Backtest complete for %s with ending equity %.2f", bot_name, result.metrics["ending_equity"])
        return result

    def run_dry(self, bot_name: str):
        bot = create_bot(bot_name, params={"history": self._demo_history(bot_name)})
        snapshots = self._demo_snapshots(bot_name)
        latest = snapshots[-1]
        decisions = bot.evaluate(latest, self._empty_portfolio())
        executor = DryRunExecutor()
        fills = executor.execute(decisions, {symbol: bar.close for symbol, bar in latest.bars.items()})
        return {"bot": bot_name, "decisions": [d.__dict__ for d in decisions], "fills": [f.__dict__ for f in fills]}

    @staticmethod
    def _empty_portfolio():
        from trading_system.core.contracts import PortfolioState

        return PortfolioState(cash=100000.0, equity=100000.0)

    @staticmethod
    def _demo_history(bot_name: str):
        base = {
            "SPY": [100, 101, 102, 103, 104, 105],
            "QQQ": [100, 101, 102, 103, 102, 104],
            "AAPL": [100, 101, 104, 107, 109, 111],
            "MSFT": [100, 100.5, 101, 102, 103, 104],
            "NVDA": [100, 103, 106, 110, 112, 115],
            "AMZN": [100, 99, 98, 101, 102, 103],
            "META": [100, 102, 103, 101, 102, 104],
            "TSLA": [100, 99, 98, 97, 96, 95],
            "NFLX": [100, 101, 99, 98, 100, 101],
            "GOOGL": [100, 102, 104, 105, 106, 108],
        }
        if bot_name == "intraday_mean_reversion":
            base["AAPL"] = [100, 100, 100, 99.8, 99.5, 97.8]
        return base

    @staticmethod
    def _demo_snapshots(bot_name: str):
        start = datetime(2026, 1, 1, 9, 30)
        history = BotManager._demo_history(bot_name)
        snapshots = []
        symbols = list(history.keys())
        for idx in range(3, 6):
            bars = {}
            for symbol in symbols:
                price = history[symbol][idx]
                bars[symbol] = Bar(symbol=symbol, timestamp=start + timedelta(days=idx), open=price - 1, high=price + 1, low=price - 2, close=price, volume=1000000)
            events = []
            if bot_name == "pead_drift" and idx == 4:
                events.append(Event(symbol="NVDA", timestamp=start + timedelta(days=idx), event_type="earnings_surprise", payload={"surprise": 0.15, "reaction": 0.04}))
            snapshots.append(MarketSnapshot(timestamp=start + timedelta(days=idx), bars=bars, events=events))
        return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading prototype bot manager")
    parser.add_argument("--list", action="store_true", help="List official prototype bots")
    parser.add_argument("--backtest", type=str, help="Run demo backtest for one bot")
    parser.add_argument("--backtest-all", action="store_true", help="Run demo backtest for all official bots")
    parser.add_argument("--dry-run", type=str, help="Generate decisions/fills without broker execution")
    args = parser.parse_args()

    manager = BotManager()
    if args.list:
        for name in manager.list_bots():
            print(name)
    elif args.backtest:
        result = manager.run_backtest(args.backtest)
        print(json.dumps(leaderboard_snapshot([result]), indent=2))
    elif args.backtest_all:
        results = [manager.run_backtest(name) for name in manager.list_bots()]
        print(json.dumps(leaderboard_snapshot(results), indent=2))
    elif args.dry_run:
        print(json.dumps(manager.run_dry(args.dry_run), indent=2, default=str))
    else:
        parser.print_help()
