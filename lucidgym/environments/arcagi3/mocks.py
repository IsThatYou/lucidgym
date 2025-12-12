"""Reusable mocks for ARC-AGI-3 transports used in tests and examples."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Deque

from .client import TransportFn


class StaticArcTransport:
    """
    Deterministic transport that replays pre-recorded ARC responses.
    """

    def __init__(self, session_data: dict[str, Any]) -> None:
        self._reset_frame = session_data["reset"]
        self._step_queue: Deque[dict[str, Any]] = deque(session_data.get("steps", []))
        self._scorecard = session_data.get("scorecard", {"cards": {}})
        self._open_card_id = self._scorecard.get("card_id", "mock-card")

    def __call__(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        if path.endswith("/RESET"):
            return deepcopy(self._reset_frame)
        if path.startswith("/api/cmd/"):
            if not self._step_queue:
                return deepcopy(self._reset_frame)
            expected = self._step_queue.popleft()
            return deepcopy(expected["response"])
        if path == "/api/scorecard/open":
            return {"card_id": self._open_card_id}
        if path == "/api/scorecard/close":
            return deepcopy(self._scorecard)
        if path.startswith("/api/scorecard/"):
            return deepcopy(self._scorecard)
        raise ValueError(f"Unexpected mock transport path: {path}")


__all__ = ["StaticArcTransport", "TransportFn"]
