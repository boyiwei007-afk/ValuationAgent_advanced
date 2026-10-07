from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol


@dataclass
class HistorySnapshot:
    raw: bytes
    source_url: str
    blocks: list[dict]
    accepted_records: int
    excluded_records: int
    warnings: list[str] = field(default_factory=list)
    catalog: dict = field(default_factory=dict)


class HistoryProvider(Protocol):
    provider_id: str
    version: str
    history_statements: tuple[str, ...]

    def fetch_history(self, ticker: str, statement: str, years: list[int], cutoff: date,
                      check_cancel: Callable[[], None]) -> HistorySnapshot: ...
