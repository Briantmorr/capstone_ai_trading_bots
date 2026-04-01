# Alpaca Paper Trading Prototype

This repo is being migrated from a class project bot collection into a spec-driven trading prototype.

## Current direction

The old bot scripts are still present as reference material, but the active prototype now targets:
- 3 official bots
- one shared Python framework
- backtest and paper-trading modes through the same bot contract
- Alpaca-backed attribution and leaderboard snapshots
- risk controls outside strategy logic

Read first:
- `docs/SALVAGE_STATUS.md`
- `docs/ARCHITECTURE.md`
- `config/bots.example.yaml`

## Official prototype bots

1. `momentum_volatility`
2. `pead_drift`
3. `intraday_mean_reversion`

## Repo status

Implemented in this migration step:
- shared strategy contract
- bot registry
- risk manager
- CSV-backed historical data loader for backtests
- Alpaca historical bar sync path that writes local CSVs for the supported universe
- causal snapshot builder with prior-bar history only
- backtest engine with next-bar-open fill semantics
- dry-run execution adapter
- Alpaca paper execution adapter with stdlib REST fallback when `alpaca-py` is unavailable
- SQLite attribution store keyed by client/broker order ids
- paper-mode reconciliation of account, orders, and positions into a bot-scoped portfolio view
- persistent per-run artifacts under `artifacts/<run_id>/`
- versioned leaderboard snapshot generator
- architecture and salvage docs
- regression tests for no-lookahead, artifacts, attribution, sync, and reconciliation semantics

Still to do:
- richer real strategy logic and validation against the written spec
- fuller live leaderboard rollup across all bots from shared account state
- production leaderboard UI integration
- PEAD event/news ingestion beyond fixture/demo events
- install/runtime validation for `alpaca-py` in environments that will place paper trades

## Setup

```bash
python3 -m unittest tests.test_framework
```

If you have `pipenv` available in the target environment, the intended workflow is still:

```bash
pipenv install --dev
pipenv run pytest
```

## CLI

List official bots:

```bash
python bot_manager.py --list
```

Sync real historical bars from Alpaca into the local CSV store:

```bash
python bot_manager.py --sync-historical SPY QQQ AAPL MSFT NVDA --start 2025-01-01
```

Run demo backtest for one bot:

```bash
python bot_manager.py --backtest momentum_volatility
```

Run demo backtests for all official bots:

```bash
python bot_manager.py --backtest-all
```

Generate dry-run decisions for one bot:

```bash
python bot_manager.py --dry-run pead_drift
```

Submit current decisions to Alpaca paper trading and persist reconciliation output:

```bash
python bot_manager.py --paper momentum_volatility
```

## Config and secrets

Use environment variables for broker/API credentials.
Do not commit secrets.
A sample non-secret config lives at `config/bots.example.yaml`.

## PEAD status

PEAD remains fixture/event-demo driven in this pass. The clean interface is in place, but a production-grade earnings/events source still needs provider selection and runtime credentials. That gap is documented rather than papered over.

## Legacy code

Legacy strategy folders remain temporarily for salvage/reference:
- `trading_bot_llm_sentiment_brian`
- `momentum_bot_carlo`
- `trading_bot_macd_melissa`
- `momentum_ml_carlo`

These are not the target architecture.
