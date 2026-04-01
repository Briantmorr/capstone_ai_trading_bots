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
- deterministic demo backtest engine
- dry-run execution adapter
- leaderboard snapshot generator
- architecture and salvage docs
- regression tests for core semantics

Still to do:
- real historical data loader
- real Alpaca paper execution orchestration
- persistent attribution store keyed by Alpaca order id
- artifact persistence
- production leaderboard UI integration
- richer strategy logic and validation against the written spec

## Setup

```bash
pipenv install --dev
pipenv run pytest
```

## CLI

List official bots:

```bash
python bot_manager.py --list
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

## Config and secrets

Use environment variables for broker/API credentials.
Do not commit secrets.
A sample non-secret config lives at `config/bots.example.yaml`.

## Legacy code

Legacy strategy folders remain temporarily for salvage/reference:
- `trading_bot_llm_sentiment_brian`
- `momentum_bot_carlo`
- `trading_bot_macd_melissa`
- `momentum_ml_carlo`

These are not the target architecture.
