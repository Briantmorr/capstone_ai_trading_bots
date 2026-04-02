from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List, Optional

from trading_system.core.contracts import Event


class EventProvider(ABC):
    @abstractmethod
    def get_events(self, bot_name: str, symbols: Iterable[str], as_of: Optional[datetime] = None) -> List[Event]:
        raise NotImplementedError


class StaticEventProvider(EventProvider):
    def __init__(self, events: Optional[List[Event]] = None):
        self.events = events or []

    def get_events(self, bot_name: str, symbols: Iterable[str], as_of: Optional[datetime] = None) -> List[Event]:
        allowed = set(symbols)
        output = [event for event in self.events if event.symbol in allowed]
        if as_of is not None:
            output = [event for event in output if event.timestamp <= as_of]
        return output


class PEADEventProvider(EventProvider):
    """
    Thin seam for future PEAD provider integration.

    Current status:
    - supports fixture/static events for deterministic tests and local dry-runs
    - leaves live provider selection outside strategy code
    """

    def __init__(self, fallback_events: Optional[List[Event]] = None):
        self._fallback = StaticEventProvider(fallback_events or [])

    def get_events(self, bot_name: str, symbols: Iterable[str], as_of: Optional[datetime] = None) -> List[Event]:
        if bot_name != "pead_drift":
            return []
        return self._fallback.get_events(bot_name=bot_name, symbols=symbols, as_of=as_of)
