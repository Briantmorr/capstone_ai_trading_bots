from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from trading_system.core.contracts import Bar


@dataclass
class AlpacaMarketDataConfig:
    key_env: str = "ALPACA_API_KEY"
    secret_env: str = "ALPACA_API_SECRET"
    base_url: str = "https://data.alpaca.markets"
    timeframe: str = "1Day"
    adjustment: str = "raw"
    feed: str = "iex"
    page_limit: int = 1000


class AlpacaHistoricalDataSource:
    def __init__(self, config: Optional[AlpacaMarketDataConfig] = None):
        self.config = config or AlpacaMarketDataConfig()

    def sync_bars(
        self,
        symbols: Iterable[str],
        output_dir: Path | str,
        start: str,
        end: Optional[str] = None,
    ) -> Dict[str, int]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}
        for symbol in symbols:
            bars = self.fetch_bars(symbol=symbol, start=start, end=end)
            self.write_csv(symbol, bars, output_path)
            counts[symbol] = len(bars)
        return counts

    def fetch_bars(self, symbol: str, start: str, end: Optional[str] = None) -> List[Bar]:
        params = {
            "symbols": symbol,
            "timeframe": self.config.timeframe,
            "start": _normalize_iso(start),
            "adjustment": self.config.adjustment,
            "feed": self.config.feed,
            "limit": str(self.config.page_limit),
        }
        if end:
            params["end"] = _normalize_iso(end)

        url = f"{self.config.base_url}/v2/stocks/bars?{urllib.parse.urlencode(params)}"
        payload = self._request_json(url)
        raw_bars = payload.get("bars", {}).get(symbol, [])
        bars = [
            Bar(
                symbol=symbol,
                timestamp=_parse_bar_timestamp(item["t"]),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item.get("v", 0.0)),
            )
            for item in raw_bars
        ]
        bars.sort(key=lambda item: item.timestamp)
        return bars

    def write_csv(self, symbol: str, bars: List[Bar], output_dir: Path | str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / f"{symbol}.csv"
        lines = ["timestamp,open,high,low,close,volume"]
        for bar in bars:
            lines.append(
                f"{bar.timestamp.strftime('%Y-%m-%d')},{bar.open},{bar.high},{bar.low},{bar.close},{int(bar.volume) if float(bar.volume).is_integer() else bar.volume}"
            )
        path.write_text("\n".join(lines) + "\n")
        return path

    def _request_json(self, url: str) -> Dict:
        key = os.environ.get(self.config.key_env)
        secret = os.environ.get(self.config.secret_env)
        if not key or not secret:
            raise RuntimeError("Missing Alpaca credentials in environment")
        request = urllib.request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request) as response:  # nosec - controlled HTTPS endpoint
            return json.loads(response.read().decode("utf-8"))


def _normalize_iso(value: str) -> str:
    if "T" in value:
        return value if value.endswith("Z") else value + "Z"
    return value + "T00:00:00Z"


def _parse_bar_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
