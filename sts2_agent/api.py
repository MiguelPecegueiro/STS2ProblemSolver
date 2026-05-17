"""HTTP client for the STS2MCP localhost REST API."""

from __future__ import annotations

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:15526"
SINGLEPLAYER_PATH = "/api/v1/singleplayer"


class STS2APIError(Exception):
    """Raised when the API returns an unexpected or error response."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class STS2Client:
    """Thin wrapper around GET/POST for singleplayer game state and actions."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}{SINGLEPLAYER_PATH}"

    def get_state(self) -> dict:
        """Fetch the current game state JSON."""
        try:
            response = self._session.get(self._endpoint, timeout=self.timeout)
        except requests.RequestException as exc:
            raise STS2APIError(f"GET failed: {exc}") from exc
        return self._parse_response(response)

    def send_action(self, action: dict) -> dict:
        """POST an action dict; returns the updated state (or API acknowledgment)."""
        try:
            response = self._session.post(
                self._endpoint, json=action, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise STS2APIError(f"POST failed: {exc}") from exc
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: requests.Response) -> dict:
        if not response.ok:
            raise STS2APIError(
                f"HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise STS2APIError("Response is not valid JSON", body=response.text) from exc
        if not isinstance(data, dict):
            raise STS2APIError(f"Expected JSON object, got {type(data).__name__}")
        status = str(data.get("status") or "").lower()
        if status == "error":
            message = data.get("message") or data.get("error") or "unknown error"
            raise STS2APIError(str(message), body=response.text)
        return data
