# Runtime Modes: Local Runner vs GitHub Actions

Status: active
Date: 2026-04-01

## Why this split exists

The old repo used GitHub Actions as the main scheduler and pushed leaderboard state outward.
The migrated prototype should separate:
- **local execution authority** for anything needing credentials, broker state, debugging, or iterative development
- **GitHub Actions** for reproducible CI/backtests and optional low-risk scheduled jobs

## Run locally when

Use local execution for:
- paper trading against Alpaca
- anything using real API credentials
- historical sync during development
- dry-runs while iterating on strategy logic
- debugging attribution/reconciliation issues
- validating shared-account PnL estimates

Recommended local commands:

```bash
python3 -m unittest tests.test_framework
python bot_manager.py --list
python bot_manager.py --backtest momentum_volatility
python bot_manager.py --dry-run pead_drift
python bot_manager.py --paper momentum_volatility
```

## Run in GitHub Actions when

Use GitHub Actions for:
- unit/regression tests
- deterministic backtests on fixture/local CSV data
- lint/test checks for PRs
- optional scheduled backtests that do not require brokerage writes

Good candidates for Actions jobs:
- `python3 -m unittest tests.test_framework`
- `python bot_manager.py --backtest-all`

## Avoid GitHub Actions for

Avoid Actions as the primary place for:
- live or paper order submission unless Brian explicitly wants that operational model
- secrets-heavy PEAD/news providers during early integration
- writing leaderboard truth to an external endpoint
- legacy `--run` / `--run-all` bot-manager paths tied to archived bots

## Current repo status

The checked-in scheduled workflows are **legacy**:
- they still reference old bot names
- they call old CLI flags not present in the migrated manager
- they still post to the external Vercel leaderboard updater

Treat them as migration leftovers, not current operating guidance.

## Recommended near-term operating model

### Local runner
- local machine owns Alpaca credentials
- local machine runs `--paper` and `--sync-historical`
- local artifacts are the source for UI/inspection

### GitHub Actions
- PR CI runs tests/backtests only
- optional nightly job can run `--backtest-all` and upload artifacts
- no external leaderboard push

## If scheduling paper mode later

If Brian wants unattended paper trading later, prefer one of:
1. a local scheduler on the trusted machine
2. a small dedicated runner/VPS with explicit secrets management

Do that only after:
- bot logic is stable
- attribution is trusted enough
- PEAD provider choice is locked
- kill-switch behavior is validated
