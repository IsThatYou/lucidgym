"""
HTTP client for the ARC-AGI-3 evaluation service.

This variant lives alongside the LucidGym environment so callers no longer need
to import the legacy ``lucidgym.integrations`` package.
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import requests
from pydantic import ValidationError
from requests import Response
from requests.cookies import RequestsCookieJar

from .structs import FrameData, GameAction, Scorecard, normalize_action_payload

logger = logging.getLogger(__name__)

TransportFn = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


class ArcAgi3ClientError(RuntimeError):
    """Raised when the ARC backend returns an error payload or invalid data."""


class ArcAgi3TransportError(ArcAgi3ClientError):
    """Raised when the underlying HTTP transport fails."""


class ArcAgi3Client:
    """
    Thin HTTP wrapper that handles authentication, cookies, and payload parsing.
    """

    def __init__(
        self,
        root_url: str | None = None,
        default_card_id: str | None = None,
        *,
        api_key: str | None = None,
        default_game_id: str | None = None,
        timeout: float = 30.0,
        cookies: Mapping[str, str] | RequestsCookieJar | None = None,
        session: requests.Session | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        self.root_url = (root_url or os.getenv("ARC_ROOT_URL", "")).rstrip("/")
        if not self.root_url:
            raise ValueError("ArcAgi3Client requires a root_url (set ARC_ROOT_URL or pass root_url).")
        self.api_key = api_key or os.getenv("ARC_API_KEY")
        self.default_card_id = default_card_id
        self.default_game_id = default_game_id
        self.timeout = timeout
        self._transport = transport

        self._session: requests.Session | None = None
        if transport is None:
            self._session = session or requests.Session()
            self._session.headers.setdefault("Accept", "application/json")
            if self.api_key:
                self._session.headers["X-API-Key"] = self.api_key
            if cookies:
                jar = self._coerce_cookies(cookies)
                self._session.cookies.update(jar)
        elif session is not None:
            raise ValueError("Specify either a custom transport or a requests.Session, not both.")

    def _coerce_cookies(self, cookies: Mapping[str, str] | RequestsCookieJar) -> RequestsCookieJar:
        if isinstance(cookies, RequestsCookieJar):
            return cookies
        jar = RequestsCookieJar()
        for key, value in cookies.items():
            jar.set(key, value)
        return jar

    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        card_id: str | None = None,
        game_id: str | None = None,
        reasoning: Any | None = None,
        guid: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> FrameData:
        """
        Issue the RESET command to the ARC backend.
        """
        if not (card_id or self.default_card_id):
            self.default_card_id = self.open_scorecard(tags=tags)
        if card_id:
            self.default_card_id = card_id
        action = GameAction.RESET
        return self._command(action, card_id=self.default_card_id, game_id=game_id, reasoning=reasoning, guid=guid)

    def step(
        self,
        action: GameAction | str,
        *,
        game_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        reasoning: Any | None = None,
        guid: str | None = None,
    ) -> FrameData:
        """
        Send a gameplay action and return the parsed frame.
        """
        action_obj = action if isinstance(action, GameAction) else GameAction.from_name(str(action))
        return self._command(
            action_obj,
            card_id=self.default_card_id,
            game_id=game_id,
            payload=payload,
            reasoning=reasoning,
            guid=guid,
        )

    def scorecard(self, *, game_id: str | None = None) -> Scorecard:
        game = game_id or self.default_game_id
        if not game:
            raise ValueError("scorecard requires game_id.")
        data = self._request_json("GET", f"/api/scorecard/{self.default_card_id}/{game}")
        try:
            return Scorecard.model_validate(data)
        except ValidationError as exc:
            raise ArcAgi3ClientError("Invalid scorecard payload from ARC backend.") from exc

    def open_scorecard(self, *, tags: Sequence[str] | None = None) -> str:
        payload = {"tags": list(tags or [])}
        print(f"Opening scorecard with tags: {payload}")
        data = self._request_json("POST", "/api/scorecard/open", payload)
        card_id = data.get("card_id")
        return card_id

    def close_scorecard(self, card_id: str) -> Scorecard | None:
        payload = {"card_id": card_id}
        try:
            data = self._request_json("POST", "/api/scorecard/close", payload)
        except ArcAgi3ClientError as exc:
            logger.warning("Failed to close scorecard %s: %s", card_id, exc)
            return None
        try:
            return Scorecard.model_validate(data)
        except ValidationError:
            logger.warning("Scorecard close returned invalid payload for %s", card_id)
            return None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()

    # ------------------------------------------------------------------ #
    def _command(
        self,
        action: GameAction,
        *,
        game_id: str | None,
        payload: Mapping[str, Any] | None = None,
        reasoning: Any | None = None,
        guid: str | None = None,
        card_id: str | None = None,
    ) -> FrameData:
        card = card_id or self.default_card_id
        game = game_id or self.default_game_id
        if not card:
            raise ValueError("card_id is required for ARC commands.")
        if not game:
            raise ValueError("game_id is required for ARC commands.")

        data = dict(payload or {})
        data.setdefault("game_id", game)
        if guid:
            data["guid"] = guid
        if reasoning is not None:
            data["reasoning"] = reasoning
        if action is GameAction.RESET:
            data.setdefault("card_id", card)
        normalized_payload = normalize_action_payload(action, data)
        
        response = self._request_json("POST", f"/api/cmd/{action.name}", normalized_payload)

        try:
            return FrameData.model_validate(response)
        except ValidationError as exc:
            raise ArcAgi3ClientError("Invalid frame payload from ARC backend.") from exc

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._transport:
            data = self._transport(method, path, deepcopy(payload) if payload else None)
        else:
            assert self._session is not None
            url = f"{self.root_url}{path}"
            try:
                if method.upper() == "GET":
                    response = self._session.get(url, timeout=self.timeout)
                else:
                    response = self._session.request(method.upper(), url, json=payload or {}, timeout=self.timeout)
            except requests.RequestException as exc:
                raise ArcAgi3TransportError(f"Transport error during {method} {path}: {exc}") from exc
            data = self._parse_response(response)

        if isinstance(data, dict) and data.get("error"):
            raise ArcAgi3ClientError(f"ARC backend reported error: {data['error']}")
        if not isinstance(data, dict):
            raise ArcAgi3ClientError("ARC backend returned a non-dict payload.")
        return data

    def _parse_response(self, response: Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ArcAgi3TransportError(f"HTTP {response.status_code}: {response.text}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ArcAgi3ClientError("ARC backend returned invalid JSON.") from exc
        return data


__all__ = [
    "ArcAgi3Client",
    "ArcAgi3ClientError",
    "ArcAgi3TransportError",
    "TransportFn",
]
