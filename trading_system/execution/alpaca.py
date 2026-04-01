from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from trading_system.core.contracts import Decision, Fill, OrderSide, PortfolioState, Position
from trading_system.storage.attribution import AttributionRecord, AttributionStore

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
except ImportError:  # pragma: no cover
    TradingClient = None
    AlpacaOrderSide = None
    TimeInForce = None
    GetOrdersRequest = None
    MarketOrderRequest = None


@dataclass
class AlpacaConfig:
    key_env: str = "ALPACA_API_KEY"
    secret_env: str = "ALPACA_API_SECRET"
    paper: bool = True
    dry_run: bool = False
    attribution_db_path: str = "artifacts/attribution.sqlite3"
    trading_base_url: str = "https://paper-api.alpaca.markets"


class DryRunExecutor:
    def __init__(self, attribution_store: Optional[AttributionStore] = None, run_id: str = "dry-run"):
        self.attribution_store = attribution_store
        self.run_id = run_id

    def execute(self, decisions: List[Decision], latest_prices: dict) -> List[Fill]:
        fills = []
        for decision in decisions:
            client_order_id = f"{decision.bot_name}-{uuid.uuid4().hex[:12]}"
            if self.attribution_store is not None:
                self.attribution_store.record(
                    AttributionRecord(
                        run_id=self.run_id,
                        bot_name=decision.bot_name,
                        symbol=decision.symbol,
                        broker="dry-run",
                        submitted_at=decision.timestamp,
                        client_order_id=client_order_id,
                        broker_order_id=client_order_id,
                        status="filled",
                        side=decision.side.value,
                        qty=decision.qty,
                        metadata=decision.metadata,
                    )
                )
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=latest_prices[decision.symbol],
                    metadata={**decision.metadata, "executor": "dry-run", "bot_name": decision.bot_name, "client_order_id": client_order_id},
                )
            )
        return fills


class AlpacaPaperExecutor:
    def __init__(self, config: AlpacaConfig, attribution_store: Optional[AttributionStore] = None, run_id: str = "paper"):
        self.config = config
        self.run_id = run_id
        self.attribution_store = attribution_store
        self._sdk_client = None
        key = os.environ.get(config.key_env)
        secret = os.environ.get(config.secret_env)
        if not key or not secret:
            raise RuntimeError("Missing Alpaca credentials in environment")
        self._key = key
        self._secret = secret
        if TradingClient is not None:
            self._sdk_client = TradingClient(key, secret, paper=config.paper)

    def get_account(self):
        if self._sdk_client is not None:
            return self._sdk_client.get_account()
        return self._request_json("GET", "/v2/account")

    def get_positions(self):
        if self._sdk_client is not None:
            return self._sdk_client.get_all_positions()
        return self._request_json("GET", "/v2/positions")

    def get_orders(self, status: str = "all"):
        if self._sdk_client is not None and GetOrdersRequest is not None:
            return self._sdk_client.get_orders(filter=GetOrdersRequest(status=status))
        return self._request_json("GET", "/v2/orders", params={"status": status, "direction": "desc", "limit": "500"})

    def execute(self, decisions: List[Decision]) -> List[Fill]:
        fills = []
        for decision in decisions:
            client_order_id = f"{decision.bot_name}-{uuid.uuid4().hex[:12]}"
            placed = self._submit_market_order(decision, client_order_id)
            self._record_submission(decision, client_order_id, placed)
            fills.append(
                Fill(
                    symbol=decision.symbol,
                    timestamp=decision.timestamp,
                    side=decision.side,
                    qty=decision.qty,
                    price=float(_get_attr(placed, "filled_avg_price", 0.0) or 0.0),
                    metadata={
                        **decision.metadata,
                        "broker": "alpaca",
                        "bot_name": decision.bot_name,
                        "client_order_id": client_order_id,
                        "alpaca_order_id": str(_get_attr(placed, "id", "")),
                        "alpaca_status": str(_get_attr(placed, "status", "submitted")),
                    },
                )
            )
        return fills

    def reconcile_state(self, bot_name: str, universe: List[str], run_id: Optional[str] = None) -> Dict:
        account = _normalize_account(self.get_account())
        positions = [_normalize_position(item) for item in self.get_positions()]
        orders = [_normalize_order(item) for item in self.get_orders(status="all")]
        attributed = self.attribution_store.list_for_run(run_id or self.run_id) if self.attribution_store is not None else []
        attributed_ids = {row["client_order_id"] for row in attributed}
        attributed_symbols = {row["symbol"] for row in attributed}
        relevant_positions = [row for row in positions if row["symbol"] in universe and row["symbol"] in attributed_symbols]
        relevant_orders = [row for row in orders if row.get("client_order_id") in attributed_ids or row.get("symbol") in universe]
        open_orders = [row for row in relevant_orders if row.get("status") not in {"filled", "canceled", "expired", "rejected"}]
        filled_orders = [row for row in relevant_orders if row.get("status") == "filled"]
        exposure = sum(abs(row.get("market_value", 0.0)) for row in relevant_positions)
        return {
            "account": account,
            "bot": {
                "bot_name": bot_name,
                "run_id": run_id or self.run_id,
                "universe": universe,
                "attributed_order_count": len(attributed),
                "reconciled_order_count": len(relevant_orders),
                "filled_order_count": len(filled_orders),
                "open_order_count": len(open_orders),
                "open_position_count": len(relevant_positions),
                "gross_exposure": exposure,
                "positions": relevant_positions,
                "orders": relevant_orders,
                "attribution": attributed,
            },
        }

    def portfolio_state_from_reconciliation(self, bot_name: str, universe: List[str], run_id: Optional[str] = None) -> PortfolioState:
        payload = self.reconcile_state(bot_name=bot_name, universe=universe, run_id=run_id)
        account = payload["account"]
        positions = {
            row["symbol"]: Position(
                symbol=row["symbol"],
                qty=row["qty"],
                avg_price=row["avg_entry_price"],
                market_price=row["market_price"],
            )
            for row in payload["bot"]["positions"]
        }
        gross_exposure = payload["bot"]["gross_exposure"]
        equity = account.get("equity", 0.0)
        cash = max(equity - gross_exposure, 0.0)
        return PortfolioState(cash=cash, equity=equity, positions=positions)

    def _submit_market_order(self, decision: Decision, client_order_id: str):
        if self._sdk_client is not None and MarketOrderRequest is not None:
            order = MarketOrderRequest(
                symbol=decision.symbol,
                qty=decision.qty,
                side=AlpacaOrderSide(decision.side.value),
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
            return self._sdk_client.submit_order(order_data=order)
        payload = {
            "symbol": decision.symbol,
            "qty": str(decision.qty),
            "side": decision.side.value,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        return self._request_json("POST", "/v2/orders", payload=payload)

    def _record_submission(self, decision: Decision, client_order_id: str, placed) -> None:
        if self.attribution_store is not None:
            self.attribution_store.record(
                AttributionRecord(
                    run_id=self.run_id,
                    bot_name=decision.bot_name,
                    symbol=decision.symbol,
                    broker="alpaca",
                    submitted_at=decision.timestamp,
                    client_order_id=client_order_id,
                    broker_order_id=str(_get_attr(placed, "id", "")),
                    status=str(_get_attr(placed, "status", "submitted")),
                    side=decision.side.value,
                    qty=decision.qty,
                    metadata=decision.metadata,
                )
            )

    def _request_json(self, method: str, path: str, params: Optional[Dict[str, str]] = None, payload: Optional[Dict] = None):
        url = self.config.trading_base_url.rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = None
        headers = {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request) as response:  # nosec - controlled HTTPS endpoint
            raw = response.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)


def _get_attr(item, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _normalize_account(item) -> Dict:
    return {
        "id": str(_get_attr(item, "id", "")),
        "status": str(_get_attr(item, "status", "")),
        "currency": str(_get_attr(item, "currency", "USD")),
        "equity": float(_get_attr(item, "equity", 0.0) or 0.0),
        "cash": float(_get_attr(item, "cash", 0.0) or 0.0),
        "buying_power": float(_get_attr(item, "buying_power", 0.0) or 0.0),
        "portfolio_value": float(_get_attr(item, "portfolio_value", _get_attr(item, "equity", 0.0)) or 0.0),
    }


def _normalize_position(item) -> Dict:
    return {
        "symbol": str(_get_attr(item, "symbol", "")),
        "qty": float(_get_attr(item, "qty", 0.0) or 0.0),
        "avg_entry_price": float(_get_attr(item, "avg_entry_price", 0.0) or 0.0),
        "market_price": float(_get_attr(item, "current_price", _get_attr(item, "market_price", 0.0)) or 0.0),
        "market_value": float(_get_attr(item, "market_value", 0.0) or 0.0),
        "unrealized_pl": float(_get_attr(item, "unrealized_pl", 0.0) or 0.0),
        "side": str(_get_attr(item, "side", "long")),
    }


def _normalize_order(item) -> Dict:
    side = str(_get_attr(item, "side", "buy"))
    filled_qty = float(_get_attr(item, "filled_qty", 0.0) or 0.0)
    qty = float(_get_attr(item, "qty", 0.0) or 0.0)
    filled_avg_price = float(_get_attr(item, "filled_avg_price", 0.0) or 0.0)
    return {
        "id": str(_get_attr(item, "id", "")),
        "client_order_id": str(_get_attr(item, "client_order_id", "")),
        "symbol": str(_get_attr(item, "symbol", "")),
        "status": str(_get_attr(item, "status", "")),
        "side": side,
        "qty": qty,
        "filled_qty": filled_qty,
        "filled_avg_price": filled_avg_price,
        "notional_filled": filled_qty * filled_avg_price,
    }
