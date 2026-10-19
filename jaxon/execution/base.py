"""Adapter interfaces. Strategy code never imports a concrete adapter.

v1 ships only CSV market data and a paper execution model. A future broker
integration implements ExecutionAdapter; a live feed implements
MarketDataAdapter. Neither exists yet — see docs/integration.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd

from ..models import Bar, Fill, Order


class MarketDataAdapter(ABC):
    @abstractmethod
    def bars(self) -> Iterator[Bar]:
        """Closed base bars in strict time order (ties broken by symbol)."""

    def is_stale(self, now: pd.Timestamp) -> bool:
        """Live feeds: True if the newest bar is older than allowed. CSV replay: never."""
        return False


class ExecutionAdapter(ABC):
    @abstractmethod
    def submit(self, order: Order) -> None: ...

    @abstractmethod
    def cancel(self, order_id: str, reason: str = "") -> None: ...

    @abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]: ...

    @abstractmethod
    def process_bar(self, bar: Bar) -> list[Fill]:
        """Paper model: decide fills for this bar. A real broker would instead
        report fills it received; it must never be asked to invent them."""


class LiveBrokerNotImplemented(ExecutionAdapter):
    """Placeholder that refuses to do anything. Live order submission is out of scope for v1."""

    def _no(self, *a, **k):
        raise NotImplementedError("Live order submission is not implemented in JAXON v1 (paper only).")

    submit = cancel = open_orders = process_bar = _no
