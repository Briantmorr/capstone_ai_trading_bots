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
  - deterministic replay engine with configurable fees/slippage
- `trading_system/execution/alpaca.py`
  - Alpaca paper adapter and a dry-run executor
- `trading_system/leaderboard/snapshot.py`
  - normalized leaderboard snapshot generation
- `trading_system/strategies/*.py`
  - official prototype strategies and specs

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
