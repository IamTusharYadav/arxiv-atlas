"""HTTP client for the Atlas read-only API."""

import logging
import os
from typing import Any

import httpx

# Point the same package at a local or staging deployment by setting ATLAS_API_URL.
DEFAULT_API_URL = "https://99a0zbyk70.execute-api.us-east-1.amazonaws.com"

# The search query rides in the request URL, which httpx logs at INFO. Research queries are
# sensitive, so keep them out of the logs whatever the host's log level is.
logging.getLogger("httpx").setLevel(logging.WARNING)


class AtlasError(Exception):
    """Something a tool hands back to the model as a note instead of crashing on."""


class AtlasUnavailable(AtlasError):
    pass


class AtlasRateLimited(AtlasError):
    def __init__(self, retry_after: int | None) -> None:
        secs = f" Retry in {retry_after}s." if retry_after else ""
        super().__init__(f"Atlas is rate limiting this client.{secs}")
        self.retry_after = retry_after


class NotFound(AtlasError):
    pass


class AtlasClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ATLAS_API_URL") or DEFAULT_API_URL).rstrip("/")
        self._http = http or httpx.Client(timeout=timeout)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # The API loads an embedding model on cold start, so the first call after an idle period
        # is slow. Measured live, it surfaces either way: our timeout fires at 30s, or the API
        # gateway gives up first and returns 503 at its own ~30s ceiling. Same transient, so
        # retry once on both rather than betting on which wins the race.
        last_exc: httpx.TimeoutException | None = None
        message = "Atlas API is unavailable."
        for attempt in (1, 2):
            try:
                resp = self._http.get(self.base_url + path, params=params)
            except httpx.TimeoutException as exc:
                last_exc, message = exc, "Atlas API timed out."
                continue
            except httpx.HTTPError as exc:
                # The transport error can name the internal endpoint, so don't let it through.
                raise AtlasUnavailable("Atlas API is unavailable.") from exc
            if resp.status_code >= 500 and attempt == 1:
                continue
            return self._handle(resp)
        raise AtlasUnavailable(message) from last_exc

    @staticmethod
    def _handle(resp: httpx.Response) -> dict[str, Any]:
        if resp.status_code == 429:
            header = resp.headers.get("retry-after")
            raise AtlasRateLimited(int(header) if header and header.isdigit() else None)
        if resp.status_code == 404:
            raise NotFound("Not in the corpus.")
        if resp.status_code >= 500:
            raise AtlasUnavailable("Atlas API is unavailable.")
        if resp.status_code >= 400:
            raise AtlasError("Atlas rejected the request.")
        data: dict[str, Any] = resp.json()
        return data
