from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from trading_system.core.contracts import Bar, Event, MarketSnapshot


class CsvHistoricalDataSource:
    """Load OHLCV bars from one CSV per symbol.

    Expected columns: timestamp,open,high,low,close,volume
    Timestamp accepts ISO-8601 or YYYY-MM-DD.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load_bars(self, symbols: Iterable[str]) -> Dict[str, List[Bar]]:
        bars_by_symbol: Dict[str, List[Bar]] = {}
        for symbol in symbols:
            path = self.root / f"{symbol}.csv"
            if not path.exists():
                continue
            bars: List[Bar] = []
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    ts = _parse_timestamp(row["timestamp"])
                    bars.append(
                        Bar(
                            symbol=symbol,
                            timestamp=ts,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )
                    )
            bars.sort(key=lambda item: item.timestamp)
            bars_by_symbol[symbol] = bars
        return bars_by_symbol


def build_market_snapshots(
    bars_by_symbol: Dict[str, List[Bar]],
    events: Optional[Iterable[Event]] = None,
    lookback_bars: int = 20,
) -> List[MarketSnapshot]:
    all_timestamps = sorted({bar.timestamp for bars in bars_by_symbol.values() for bar in bars})
    event_map = defaultdict(list)
    for event in events or []:
        event_map[event.timestamp].append(event)

    history: Dict[str, List[Bar]] = defaultdict(list)
    snapshots: List[MarketSnapshot] = []
    for timestamp in all_timestamps:
        current_bars: Dict[str, Bar] = {}
        prior_bars: Dict[str, List[Bar]] = {}
        for symbol, bars in bars_by_symbol.items():
            maybe_bar = next((bar for bar in bars if bar.timestamp == timestamp), None)
            if maybe_bar is None:
                continue
            current_bars[symbol] = maybe_bar
            prior_bars[symbol] = list(history[symbol][-lookback_bars:])
        if current_bars:
            snapshots.append(
                MarketSnapshot(
                    timestamp=timestamp,
                    bars=current_bars,
                    history=prior_bars,
                    events=list(event_map.get(timestamp, [])),
                )
            )
        for symbol, bar in current_bars.items():
            history[symbol].append(bar)
    return snapshots


def _parse_timestamp(raw: str) -> datetime:
    raw = raw.strip()
    if "T" in raw:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    return datetime.strptime(raw, "%Y-%m-%d")
