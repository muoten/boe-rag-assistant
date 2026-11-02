"""Tests for the Foundry provider and its spending allowance (no Azure calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from boe_rag.llm.base import ChatMessage, LLMError, LLMRateLimitError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.llm.foundry import FoundryProvider
from boe_rag.llm.quota import QuotaExceededError, micro

_ENDPOINT = "https://example.services.ai.azure.com"
_MESSAGES = [ChatMessage(role="user", content="¿Qué plazo?")]


@dataclass
class _Token:
    token: str


class _Credential:
    def __init__(self) -> None:
        self.calls = 0

    def get_token(self, *scopes: str) -> _Token:
        assert scopes == ("https://ai.azure.com/.default",)
        self.calls += 1
        return _Token("t")


@dataclass
class _Ledger:
    """In-memory stand-in for :class:`BlobQuota`."""

    budget: int
    spent: int = 0
    events: list[tuple[str, int]] = field(default_factory=list)

    def reserve(self, amount: int) -> str:
        if self.spent + amount > self.budget:
            raise QuotaExceededError("exhausted")
        self.spent += amount
        self.events.append(("reserve", amount))
        return "2026-11"

    def refund(self, month: str, amount: int) -> None:
        self.spent -= amount
        self.events.append(("refund", amount))


def _provider(handler: Any, ledger: _Ledger | None = None) -> FoundryProvider:
    provider = FoundryProvider(_ENDPOINT, "gpt", credential=_Credential(), quota=ledger)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def _reply(content: str | None, usage: dict[str, int] | None = None) -> httpx.Response:
    body: dict[str, Any] = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_sends_reasoning_request_without_sampling_options() -> None:
    """The deployment gets max_completion_tokens and no temperature."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _reply("Diez días [1].")

    answer = _provider(handler).complete(_MESSAGES, temperature=0.0, max_tokens=512)
    assert answer == "Diez días [1]."
    assert seen["url"] == _ENDPOINT + "/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer t"
    body = seen["body"]
    assert body["model"] == "gpt"
    assert body["max_completion_tokens"] == 512
    assert body["reasoning_effort"] == "minimal"
    assert "temperature" not in body and "max_tokens" not in body


def test_refunds_the_unused_reservation_from_reported_usage() -> None:
    """The ledger ends at the actual cost of the reported tokens."""
    ledger = _Ledger(budget=micro(1))
    provider = _provider(
        lambda _: _reply("Sí [1].", {"prompt_tokens": 1000, "completion_tokens": 50}),
        ledger,
    )
    provider.complete(_MESSAGES, max_tokens=512)
    assert ledger.spent == micro(1000 * 0.404e-6 + 50 * 3.2317e-6)
    assert [kind for kind, _ in ledger.events] == ["reserve", "refund"]


def test_missing_usage_keeps_the_whole_reservation() -> None:
    """Without usage the cost is unknown, so nothing is refunded."""
    ledger = _Ledger(budget=micro(1))
    _provider(lambda _: _reply("Sí [1]."), ledger).complete(_MESSAGES, max_tokens=512)
    assert ledger.events == [("reserve", ledger.spent)]


def test_exhausted_allowance_never_calls_the_model() -> None:
    """A reservation the budget cannot cover stops before any HTTP request."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("model must not be called")

    with pytest.raises(QuotaExceededError):
        _provider(handler, _Ledger(budget=0)).complete(_MESSAGES)


def test_exhausted_allowance_falls_through_the_chain() -> None:
    """QuotaExceededError is a rate limit, so the fallback chain moves on."""

    class _Free:
        name = "free"

        def complete(self, messages: Any, **kwargs: Any) -> str:
            return "gratis [1]"

    paid = _provider(lambda _: _reply("x"), _Ledger(budget=0))
    assert FallbackProvider([paid, _Free()]).complete(_MESSAGES) == "gratis [1]"
    assert issubclass(QuotaExceededError, LLMRateLimitError)


def test_failed_call_keeps_its_reservation() -> None:
    """A request error leaves the reservation booked."""
    ledger = _Ledger(budget=micro(1))
    provider = _provider(lambda _: httpx.Response(400, text="bad"), ledger)
    with pytest.raises(LLMError):
        provider.complete(_MESSAGES)
    assert [kind for kind, _ in ledger.events] == ["reserve"]


def test_empty_completion_is_an_error() -> None:
    """A reasoning model that used its cap thinking returns no text."""
    with pytest.raises(LLMError, match="empty Foundry completion"):
        _provider(lambda _: _reply("")).complete(_MESSAGES)


def test_estimate_covers_a_full_output() -> None:
    """The reservation bounds a call whose input averages four chars per token."""
    provider = _provider(lambda _: _reply("x"))
    long = [ChatMessage(role="user", content="a" * 12_000)]
    worst = micro(3000 * 0.404e-6 + 512 * 3.2317e-6)
    assert provider.estimate(long, 512) > worst


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.services.ai.azure.com",
        "https://example.com",
        "https://example.services.ai.azure.com/openai",
        "https://user:pw@example.services.ai.azure.com",
    ],
)
def test_rejects_unexpected_endpoints(endpoint: str) -> None:
    """Only a plain https Azure AI origin is accepted."""
    with pytest.raises(LLMError):
        FoundryProvider(endpoint, "gpt", credential=_Credential())


def test_unconfigured_foundry_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without FOUNDRY_* the chain is the one it was before."""
    for var in ("FOUNDRY_ENDPOINT", "FOUNDRY_DEPLOYMENT", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert [p.name.split(":")[0] for p in build_available_providers()] == ["groq"]


def test_configured_foundry_leads_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """With FOUNDRY_* set, Foundry is tried first."""
    monkeypatch.setenv("FOUNDRY_ENDPOINT", _ENDPOINT)
    monkeypatch.setenv("FOUNDRY_DEPLOYMENT", "gpt")
    monkeypatch.delenv("QUOTA_CONTAINER_URL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    names = [p.name for p in build_available_providers()]
    assert names[0] == "foundry:gpt"
