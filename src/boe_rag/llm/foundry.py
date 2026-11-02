"""Microsoft Foundry chat-completion provider, authenticated by managed identity.

Calls one fixed model deployment on an Azure AI Foundry resource through its
OpenAI-compatible ``/openai/v1/chat/completions`` route, with an Entra ID token
instead of an API key: on Azure the app's user-assigned managed identity
(``AZURE_CLIENT_ID``), elsewhere whatever ``DefaultAzureCredential`` finds (e.g. an
``az login`` session) for local evals.

The default deployment is a reasoning model (gpt-5-mini). Reasoning models reject
sampling options, so ``temperature`` is accepted for protocol compatibility but not
sent, and ``max_tokens`` becomes ``max_completion_tokens`` — a cap that also bounds
the hidden reasoning, kept small with ``reasoning_effort="minimal"``.

When a :class:`~boe_rag.llm.quota.SpendingLedger` is attached, every call reserves
its worst-case cost before the request and refunds the unused part from the
reported usage, so a public endpoint can never spend past its monthly allowance.

Configuration: ``FOUNDRY_ENDPOINT`` (``https://<resource>.services.ai.azure.com``)
and ``FOUNDRY_DEPLOYMENT``; ``QUOTA_CONTAINER_URL`` / ``QUOTA_MONTHLY_EUR`` enable
the allowance. Needs the ``azure`` extra.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from boe_rag.llm._http import post_json_with_retry
from boe_rag.llm.base import ChatMessage, LLMError
from boe_rag.llm.quota import SpendingLedger, micro, quota_from_env

#: Token audience for Foundry data-plane calls.
_SCOPE = "https://ai.azure.com/.default"

#: gpt-5-mini Global Standard retail prices in EUR per token (input, output),
#: Spain Central, read on 2 November 2026. Override for another deployment.
GPT5_MINI_EUR = (0.404e-6, 3.2317e-6)

#: Characters per token assumed when reserving. Spanish legal text averages about
#: four characters per token; three over-reserves, and the refund settles the rest.
_RESERVE_CHARS_PER_TOKEN = 3


class _Credential(Protocol):
    """The subset of an ``azure.identity`` credential the provider uses."""

    def get_token(self, *scopes: str) -> Any:
        """Return an access token with a ``token`` attribute."""
        ...


class FoundryProvider:
    """A fixed Foundry model deployment (OpenAI-compatible API, Entra ID auth).

    Args:
        endpoint: Foundry resource origin; falls back to ``FOUNDRY_ENDPOINT``.
        deployment: Deployment name; falls back to ``FOUNDRY_DEPLOYMENT``.
        credential: Token credential; defaults to the managed identity named by
            ``AZURE_CLIENT_ID``, else ``DefaultAzureCredential``.
        quota: Optional spending allowance; defaults to :func:`quota_from_env`.
        prices: EUR per input and output token, used for reservations and refunds.
        reasoning_effort: Sent to reasoning deployments; ``None`` omits it.
        timeout: Per-request timeout in seconds.

    Raises:
        LLMError: If the endpoint or deployment is missing or malformed.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        *,
        credential: _Credential | None = None,
        quota: SpendingLedger | None = None,
        prices: tuple[float, float] = GPT5_MINI_EUR,
        reasoning_effort: str | None = "minimal",
        timeout: float = 60.0,
    ) -> None:
        """Validate the deployment target and open an HTTP client."""
        origin = endpoint or os.environ.get("FOUNDRY_ENDPOINT")
        name = deployment or os.environ.get("FOUNDRY_DEPLOYMENT")
        if not origin or not name:
            raise LLMError(
                "Foundry not configured; set FOUNDRY_ENDPOINT and FOUNDRY_DEPLOYMENT."
            )
        parsed = urlparse(origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".services.ai.azure.com")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise LLMError("FOUNDRY_ENDPOINT must be a plain https Azure AI origin")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise LLMError("FOUNDRY_DEPLOYMENT is not a valid deployment name")
        self._url = origin.rstrip("/") + "/openai/v1/chat/completions"
        self._deployment = name
        self._credential = credential
        self._quota = quota if quota is not None else quota_from_env()
        self._prices = prices
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, trust_env=False)

    @property
    def name(self) -> str:
        """Provider identifier, e.g. ``foundry:rag-poc-gpt5mini-app``."""
        return f"foundry:{self._deployment}"

    def _token(self) -> str:
        """Fetch an Entra ID access token for the Foundry data plane."""
        if self._credential is None:
            from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

            client_id = os.environ.get("AZURE_CLIENT_ID")
            self._credential = (
                ManagedIdentityCredential(client_id=client_id)
                if client_id
                else DefaultAzureCredential()
            )
        try:
            token: str = self._credential.get_token(_SCOPE).token
        except Exception as err:
            raise LLMError("Foundry authentication failed") from err
        return token

    def estimate(self, messages: Sequence[ChatMessage], max_tokens: int) -> int:
        """Worst-case micro-euros for one call: generous input plus a full output."""
        chars = sum(len(m.content) for m in messages)
        tokens_in = chars // _RESERVE_CHARS_PER_TOKEN + 50 * len(messages)
        rate_in, rate_out = self._prices
        return micro(tokens_in * rate_in + max_tokens * rate_out)

    def charge(self, usage: object) -> int | None:
        """Actual micro-euros from reported usage; ``None`` when it is incomplete."""
        if not isinstance(usage, dict):
            return None
        tokens_in, tokens_out = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        if type(tokens_in) is not int or type(tokens_out) is not int:
            return None
        rate_in, rate_out = self._prices
        return micro(tokens_in * rate_in + tokens_out * rate_out)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion; ``temperature`` is ignored (see module docs).

        Raises:
            QuotaExceededError: If the monthly allowance cannot cover the call.
            LLMError: If authentication or the request fails, or no text returns.
        """
        del temperature  # reasoning deployments reject sampling options
        headers = {"Authorization": f"Bearer {self._token()}"}
        reserved = self.estimate(messages, max_tokens)
        month = self._quota.reserve(reserved) if self._quota else None
        body: dict[str, Any] = {
            "model": self._deployment,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        # A failed call keeps its whole reservation: the ledger errs low.
        data = post_json_with_retry(
            self._client,
            self._url,
            json=body,
            headers=headers,
            timeout=self._timeout,
        )
        if self._quota and month:
            spent = self.charge(data.get("usage"))
            if spent is not None:
                self._quota.refund(month, reserved - spent)
        return _extract_text(data)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the generated text out of a chat-completions payload."""
    try:
        choice = data["choices"][0]
        text: str = (choice["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError) as err:
        raise LLMError("unexpected Foundry response shape") from err
    if not text:
        # A reasoning model can spend the whole token cap thinking and say nothing.
        reason = choice.get("finish_reason")
        raise LLMError(f"empty Foundry completion (finish_reason={reason})")
    return text
