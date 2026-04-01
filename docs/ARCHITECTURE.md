# Trading Prototype Architecture

## Goal

Deliver a spec-driven trading prototype with 3 bots that run through the same framework in:
- backtest mode
- Alpaca paper mode

## Core principles

1. strategy code decides **what** to trade
2. execution adapters decide **how** to place/simulate trades
3. risk manager decides **whether** a proposed action is allowed
4. leaderboard derives live truth from broker state plus local attribution logs
5. research/training stays outside the live execution path

## Module layout

- `trading_system/core/contracts.py`
  - canonical types for market snapshots, decisions, orders, fills, and run results
- `trading_system/core/bot.py`
  - base strategy contract shared by backtest and paper mode
- `trading_system/core/registry.py`
  - official bot registry for the 3 prototype bots
- `trading_system/core/risk.py`
  - shared limits, kill switch, stale-data protection, daily loss cap, exposure checks
- `trading_system/core/backtest.py`
  - deterministic replay engine with configurable fees/slippage and next-bar-open fills
- `trading_system/data/market_data.py`
  - CSV-backed historical data ingestion and causal snapshot construction
- `trading_system/execution/alpaca.py`
  - Alpaca paper adapter and a dry-run executor with attribution hooks
- `trading_system/storage/attribution.py`
  - SQLite mapping between bot/run/client order ids and broker order ids
- `trading_system/reporting/artifacts.py`
  - per-run artifact persistence for metrics, decisions, trades, and snapshots
- `trading_system/leaderboard/snapshot.py`
  - normalized leaderboard snapshot generation
- `trading_system/strategies/*.py`
  - official prototype strategies and specs

## Data / replay semantics

Historical replay now follows these rules:
- bars are loaded from one CSV per symbol
- `MarketSnapshot.bars` contains the current decision bar
- `MarketSnapshot.history[symbol]` contains only bars strictly earlier than the decision bar
- strategies may inspect the current bar plus prior history, but not future bars
- approved orders fill on the next available bar open
- if the final snapshot emits a decision, it remains unfilled and is surfaced as a warning

This makes the current backtest path timestamp-safe enough for milestone validation and regression testing.

## Artifact contract

Each run writes under `artifacts/<run_id>/`:
- `metrics.json`
- `decisions.json`
- `trade_log.json`
- `leaderboard_snapshot.json`
- `run_manifest.json`

## Leaderboard snapshot contract

`leaderboard_snapshot.json` now includes:
- `contract_version`
- `generated_at`
- `run_id`
- `source`
- `account`
- `bots[]` with rank, equity, cash, trade count, halt state, warnings, and last equity point

## Official bots

### 1. momentum_volatility
- cadence: daily
- idea: rank liquid universe by trailing momentum, size by inverse volatility, cap exposures

### 2. pead_drift
- cadence: event-driven / daily event scan
- idea: react to earnings-style event surprises using delayed continuation rules, with timestamp-safe event handling

### 3. intraday_mean_reversion
- cadence: hourly
- idea: fade short-term dislocations when trend and volatility filters allow it

## Leaderboard truth model

Leaderboard rows should be reconstructible from:
- Alpaca account equity/cash
- Alpaca positions
- Alpaca orders/fills
- local bot decision log and broker-order attribution

Local artifact outputs should include:
- `artifacts/<run_id>/trade_log.json`
- `artifacts/<run_id>/decisions.json`
- `artifacts/<run_id>/metrics.json`
- `artifacts/<run_id>/leaderboard_snapshot.json`

## Backtest semantics

- engine replays bars/events in chronological order
- strategy only sees data up to the decision timestamp
- fills occur using a documented fill model
- slippage and commission are configurable
- outputs are deterministic from config + code version + inputs

## Risk model

Shared defaults:
- no shorting unless enabled per bot
- max 20% per symbol
- max 100% gross exposure
- stale-data halt
- repeated execution error kill switch
- daily loss cap per bot

## Migration note

Legacy bot scripts remain in repo as references only. New development should target the shared framework and the 3 official bot modules.
