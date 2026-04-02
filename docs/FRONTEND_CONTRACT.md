# Frontend Salvage Contract

Status: active
Date: 2026-04-01

## UI salvage finding

There is **no frontend source code in this repo**.

What exists instead:
- legacy GitHub Actions calling an external Vercel leaderboard endpoint
- backend artifact generation for `leaderboard_snapshot.json`
- paper-run reconciliation output in `paper_run.json`

That means UI salvage cannot currently mean "patch the checked-in frontend". The actionable next step is to freeze the backend contract that a recovered or rebuilt frontend can target.

## Source-of-truth files

### 1. `artifacts/<run_id>/leaderboard_snapshot.json`
Use for leaderboard table / rankings.

Top-level fields:
- `contract_version`
- `generated_at`
- `run_id`
- `source`
- `account`
- `bots[]`

Each `bots[]` row currently includes:
- `rank`
- `bot_name`
- `mode`
- `ending_equity`
- `ending_cash`
- `trade_count`
- `halted`
- `warnings`
- `last_equity_point`

### 2. `artifacts/<run_id>/paper_run.json`
Use for bot drilldown / recent orders / reconciliation.

Top-level fields:
- `bot`
- `run_id`
- `decision_count`
- `fills[]`
- `reconciliation.account`
- `reconciliation.bot.positions[]`
- `reconciliation.bot.orders[]`
- `reconciliation.bot.attribution[]`
- `reconciliation.bot.pnl_allocation`

## Minimal frontend data model

### Leaderboard page
Required columns:
- rank
- bot name
- mode
- allocated or ending equity
- cash
- trade count
- halted status
- warnings
- last update timestamp

Render priority:
1. `paper_run.json` allocation/reconciliation values when source is paper/live
2. `leaderboard_snapshot.json` metrics when source is backtest/dry-run

### Bot detail page
Required sections:
- account summary
- attributed open positions
- recent attributed orders
- estimated PnL allocation
- recent fills
- warnings / halt state

## Shared-account caveat

Paper mode is using a **shared Alpaca paper account** with local order attribution.
Per-bot PnL is therefore an approximation, not a broker-native ledger split.
Current allocation method:
- attribute filled orders by `client_order_id`
- derive bot net quantity by symbol
- allocate current position value and unrealized PnL by bot net-quantity share
- allocate account cash by exposure share

Frontend should label this clearly as:
- `Estimated shared-account allocation`

## API shape for future local service

If the frontend is rebuilt as a local or Vercel app, the minimum API should mirror the artifact contract:
- `GET /api/leaderboard/latest` -> latest `leaderboard_snapshot.json`
- `GET /api/bots/:bot_name/latest` -> latest `paper_run.json` or drilldown payload for that bot
- `GET /api/runs/:run_id` -> manifest + snapshot + trade log bundle

## Salvage recommendation

- Do **not** keep the old push-to-Vercel update endpoint as the truth source.
- Do salvage any old UI visuals/components if they can be pointed at this contract.
- If the old frontend repo is found later, adapt it to these files/API routes instead of reintroducing pushed leaderboard state.
