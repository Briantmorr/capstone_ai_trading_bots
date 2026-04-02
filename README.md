# Alpaca Paper Trading Prototype

This repo is the active backend for the LLM paper-trading prototype.
It is being migrated from a class-project bot collection into a spec-driven trading framework with a salvageable frontend contract.

## Current architecture

The active backend now centers on:
- **3 official bots**
- **one shared Python framework**
- **backtest and paper modes through the same bot contract**
- **Alpaca-backed attribution and leaderboard snapshots**
- **risk controls outside strategy logic**
- **local artifacts as the frontend integration surface**

Read first:
- `docs/SALVAGE_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/FRONTEND_CONTRACT.md`
- `docs/RUNTIME_MODES.md`
- `docs/PEAD_PROVIDER_PLAN.md`
- `config/bots.example.yaml`

## Official prototype bots

1. `momentum_volatility`
2. `pead_drift`
3. `intraday_mean_reversion`

## What exists now

Implemented in the current migration slice:
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
- shared-account PnL allocation estimate for bot-scoped paper-mode reporting
- paper-mode reconciliation of account, orders, and positions into a bot-scoped portfolio view
- persistent per-run artifacts under `artifacts/<run_id>/`
- versioned leaderboard snapshot generator
- explicit backend/frontend salvage contract docs
- runtime split guidance for local runner vs GitHub Actions
- PEAD event-provider seam plus integration planning docs
- regression tests for no-lookahead, artifacts, attribution, sync, reconciliation, and allocation semantics

## What is still missing

Still to do:
- richer final strategy logic and validation against the written spec
- fuller live leaderboard rollup across all bots from shared-account state
- production leaderboard UI integration and deployment path
- PEAD event/news ingestion beyond fixture/demo events
- operational validation of kill-switch behavior in live-like runs
- install/runtime validation of `alpaca-py` wherever paper trades will actually be placed

## Runtime model

### Local machine should own
- Alpaca credentials
- `--paper` runs
- `--sync-historical`
- dry-runs during iteration
- artifact generation that the frontend consumes

### GitHub Actions should own
- tests
- deterministic backtests
- optional artifact-producing CI jobs

### Deprecated workflow assumptions
Treat these as legacy and not current operating guidance:
- one Alpaca account per bot
- pushing leaderboard truth to an external Vercel endpoint
- legacy `--run` / `--run-all` paths tied to archived bots
- old scheduled workflows that reference deprecated CLI flags or bot names

## Storage and artifact model

### Local storage
- historical market data: local CSVs
- attribution: SQLite
- run outputs: JSON artifacts

### Artifact bundle
Each run writes under `artifacts/<run_id>/`:
- `metrics.json`
- `decisions.json`
- `trade_log.json`
- `leaderboard_snapshot.json`
- `run_manifest.json`
- `paper_run.json` for paper-mode submissions and reconciliation

These files are the current bridge to the frontend repo.

## Setup

Minimal regression pass:

```bash
python3 -m unittest tests.test_framework
```

If you use `pipenv` in this environment:

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

## UI contract status

There is no checked-in frontend in this repo.
The current backend contract for any salvaged or rebuilt UI is documented in:
- `docs/FRONTEND_CONTRACT.md`
- `docs/examples.leaderboard_snapshot.json`

## PEAD status

PEAD remains fixture/event-demo driven in this pass.
The clean provider interface is now in place, but a production-grade earnings/events source still needs provider selection and runtime credentials.

## Legacy reference code

Legacy strategy folders remain temporarily for salvage/reference only:
- `trading_bot_llm_sentiment_brian`
- `momentum_bot_carlo`
- `trading_bot_macd_melissa`
- `momentum_ml_carlo`

Do not treat these folders as the target architecture.
