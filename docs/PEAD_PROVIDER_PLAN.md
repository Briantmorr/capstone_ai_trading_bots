# PEAD Provider Plan

Status: active
Date: 2026-04-01

## Current state

`pead_drift` now has a clean event-provider seam.
The strategy no longer needs to know where earnings events come from.

Current implementation:
- `trading_system/data/events.py`
- `PEADEventProvider`
- fallback static event support for deterministic tests and dry-runs

## What the provider must supply

For each earnings event, the backend needs at minimum:
- `symbol`
- `timestamp`
- `event_type = earnings_surprise`
- payload fields:
  - `surprise`
  - `reaction`

Preferred additional fields:
- `period_end`
- `fiscal_quarter`
- `reported_eps`
- `expected_eps`
- `revenue_surprise`
- `session` (pre-market / post-market)
- `source`

## Provider selection notes

### Good prototype choices
1. Finnhub earnings calendar / earnings surprise style endpoints
2. Polygon or another market-data provider if Brian already has access
3. a maintained local CSV/event fixture pipeline as a temporary bridge

### Selection criteria
- event timestamp quality
- surprise/estimate availability
- sane rate limits/cost
- easy local secret management
- ability to reconstruct events in backtests without lookahead leakage

## Integration rules

When wiring a real provider:
- keep provider code in `trading_system/data/events.py` or a sibling provider module
- normalize everything into `Event`
- persist fetched raw payloads or normalized event snapshots for replay
- never let the strategy call provider APIs directly
- ensure event timestamps reflect when the market could actually know the result

## Backtest caution

PEAD is especially sensitive to lookahead leakage.
A real provider integration is not done until:
- event timestamps are trustworthy
- market session timing is encoded
- replay only exposes events at or after their true publication time

## Recommendation

Implement provider ingestion next only after the UI/backend contract and shared-account reporting are stable enough to inspect results.
That ordering is less glamorous, but it avoids building event plumbing into a moving target.
