# Salvage / Architecture Status

Status: active migration
Date: 2026-04-01

## Summary

This repo is salvageable as a **pattern library**, not as the final prototype architecture.

### Keep
- centralized orchestration idea from `bot_manager.py`
- per-bot logging idea
- Alpaca integration patterns
- some indicator and data-fetch snippets from legacy bots

### Replace
- one-account-per-bot operating model
- live-path model training/inference coupling
- external pushed leaderboard endpoint as source of truth
- bot-specific scripts with no shared interface
- weak separation between strategy, execution, risk, and research

## Current findings

### Legacy strengths
- `momentum_bot_carlo` is the cleanest execution skeleton.
- `trading_bot_llm_sentiment_brian` contains reusable news/sentiment ingestion ideas.
- `trading_bot_macd_melissa` contains indicator snippets worth extracting.
- `momentum_ml_carlo` has usable ranking/filter ideas but trains in the run path.

### Legacy blockers
- no standard bot contract
- no backtest engine shared with live mode
- no clean risk layer outside strategy logic
- no durable attribution model for mapping Alpaca orders/fills back to a bot identity
- scheduled workflows still assume external leaderboard update instead of Alpaca-derived state

## Target architecture locked for prototype

- **3 official bots only**
  - cross-sectional momentum with volatility targeting
  - PEAD event-driven bot
  - intraday mean reversion with trend/volatility filter
- **shared interfaces** for backtest and paper mode
- **single framework** with clear modules:
  - strategy contract
  - bot registry
  - risk manager
  - backtest engine
  - Alpaca paper adapter
  - attribution + leaderboard snapshot pipeline
- **leaderboard truth** comes from Alpaca account/orders/positions plus local decision logs
- **dry-run mode** supported for safe paper-trading validation

## UI recommendation

UI was not inspected in this repo because there is no frontend here. The API workflow points at an external Vercel leaderboard. For the prototype, the backend should first produce a local Alpaca-backed leaderboard snapshot contract. UI salvage/rebuild should target that contract.

## Security note

No plaintext secrets were found in tracked source during this inspection. Secrets are referenced through env vars / GitHub Actions secrets. Continue using sample config files only.

## Immediate migration plan

1. introduce shared Python package for contracts, risk, backtest, execution, and leaderboard snapshots
2. define 3 official bot configs and strategy specs in repo
3. wire `bot_manager.py` to the new registry/runner instead of direct legacy script loading
4. add regression tests for allocation, no-lookahead backtest behavior, and risk halts
5. preserve legacy bots temporarily as reference implementations only
