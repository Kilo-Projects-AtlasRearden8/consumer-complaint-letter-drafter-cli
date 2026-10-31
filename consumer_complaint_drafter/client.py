"""Small HTTP client for OpenAI-compatible chat-completion endpoints.

Uses only the Python standard library so the tool stays local-first with no
mandatory third-party dependencies and no API keys required.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class EndpointError(RuntimeError):
    """Raised when the language-model endpoint cannot be reached or fails."""


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.3
    max_tokens: int | None = None


@dataclass(frozen=True)
class CompletionResponse:
    content: str
    model: str


class CompletionClient:
    """Client for an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": request.model or self.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EndpointError(
                f"Endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(
                f"Could not reach language-model endpoint at {self.endpoint}. "
                "Is Ollama (or another OpenAI-compatible server) running? "
                f"Underlying error: {exc}"
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EndpointError(
                f"Endpoint returned invalid JSON: "
                f"{raw.decode('utf-8', errors='replace')[:200]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(
                f"Endpoint response missing expected completion content: {data!r}"
            ) from exc

        return CompletionResponse(content=str(content), model=str(data.get("model", "")))
