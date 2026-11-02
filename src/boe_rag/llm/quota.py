"""Persistent monthly spending allowance for a paid LLM provider.

A public demo backed by a pay-per-token model needs a hard ceiling that survives
restarts and holds across replicas. :class:`BlobQuota` keeps one small JSON blob per
month in an Azure Storage container (``{"used": n, "spent_micro_eur": m}``) and
updates it with ETag-conditional writes, so concurrent requests can never both spend
the last of the allowance.

Every generation first **reserves** its worst-case cost; once the provider reports
token usage the unused part is **refunded**. A call that fails, or reports no usage,
keeps its full reservation — the ledger errs on the side of spending less.

Amounts are integer micro-euros to keep the arithmetic exact. Needs the ``azure``
extra (``azure-identity`` and ``azure-storage-blob``); authentication is managed
identity only — no keys or connection strings.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

from boe_rag.llm.base import LLMError, LLMRateLimitError


class QuotaExceededError(LLMRateLimitError):
    """The monthly allowance cannot cover this generation.

    A rate-limit subclass, so a :class:`~boe_rag.llm.factory.FallbackProvider`
    cools the paid provider down and moves on to the next one, if any.
    """


class QuotaUnavailableError(LLMError):
    """The ledger could not be read or written; the call is not admitted."""


def micro(euros: float) -> int:
    """Convert euros to integer micro-euros, rounding up."""
    return math.ceil(round(euros * 1_000_000, 6))


class SpendingLedger(Protocol):
    """What a paid provider needs from an allowance."""

    def reserve(self, amount: int) -> str:
        """Reserve ``amount`` micro-euros; return the month it was booked in."""
        ...

    def refund(self, month: str, amount: int) -> None:
        """Return ``amount`` unused micro-euros to ``month`` (best effort)."""
        ...


class _Blob(Protocol):
    """The subset of ``azure.storage.blob.BlobClient`` the ledger uses."""

    def download_blob(self) -> Any:
        """Return a downloader with ``readall()`` and ``properties.etag``."""
        ...

    def upload_blob(self, data: str, **kwargs: Any) -> Any:
        """Write the blob, optionally conditioned on an ETag."""
        ...


class BlobQuota:
    """A monthly euro allowance stored in an Azure blob container.

    Args:
        container_url: ``https://<account>.blob.core.windows.net/<container>``,
            without credentials, query string or nested path.
        budget_eur: Monthly allowance in euros (0 < budget <= 10).
        client_id: User-assigned managed identity client id; falls back to
            ``AZURE_CLIENT_ID``.
        factory: Builds a blob client from a blob URL; tests inject a fake.

    Raises:
        ValueError: If the URL or the budget is out of bounds.
    """

    def __init__(
        self,
        container_url: str,
        budget_eur: float,
        *,
        client_id: str | None = None,
        factory: Callable[[str], _Blob] | None = None,
    ) -> None:
        """Validate the container URL and budget and prepare the blob factory."""
        parsed = urlparse(container_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".blob.core.windows.net")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or len(parsed.path.strip("/").split("/")) != 1
        ):
            raise ValueError("Quota storage must be an Azure container URL")
        if not 0 < budget_eur <= 10:
            raise ValueError("Monthly budget must be between 0 and 10 euros")
        self._url = container_url.rstrip("/")
        self.budget = micro(budget_eur)
        if factory is None:
            factory = _managed_identity_factory(client_id)
        self._factory = factory

    def _update(
        self, change: Callable[[dict[str, int]], dict[str, int]], month: str
    ) -> dict[str, int]:
        """Apply ``change`` to one month's state with an ETag-guarded write."""
        from azure.core import MatchConditions
        from azure.core.exceptions import (
            ResourceExistsError,
            ResourceModifiedError,
            ResourceNotFoundError,
        )

        try:
            blob = self._factory(f"{self._url}/{month}.json")
            for _ in range(5):
                try:
                    download = blob.download_blob()
                    stored = json.loads(download.readall())
                    state = {
                        "used": stored["used"],
                        "spent_micro_eur": stored["spent_micro_eur"],
                    }
                except ResourceNotFoundError:
                    with contextlib.suppress(ResourceExistsError):
                        blob.upload_blob(
                            json.dumps({"used": 0, "spent_micro_eur": 0}),
                            overwrite=False,
                        )
                    continue
                if any(type(v) is not int or v < 0 for v in state.values()):
                    raise QuotaUnavailableError("Invalid quota state")
                new = change(state)
                try:
                    blob.upload_blob(
                        json.dumps(new),
                        overwrite=True,
                        etag=download.properties.etag,
                        match_condition=MatchConditions.IfNotModified,
                    )
                except (ResourceModifiedError, ResourceExistsError):
                    continue
                return new
            raise QuotaUnavailableError("Concurrent quota updates; retry later")
        except (QuotaExceededError, QuotaUnavailableError):
            raise
        except Exception as err:
            # Never reset unknown, corrupt or unreachable state to zero.
            raise QuotaUnavailableError("Quota storage unavailable") from err

    def reserve(self, amount: int) -> str:
        """Reserve ``amount`` micro-euros in the current month.

        Raises:
            QuotaExceededError: If the reservation would pass the budget.
            QuotaUnavailableError: If the ledger cannot be updated.
        """
        if type(amount) is not int or amount < 0:
            raise ValueError("Reservations are non-negative micro-euros")
        month = datetime.now(UTC).strftime("%Y-%m")

        def change(state: dict[str, int]) -> dict[str, int]:
            if state["spent_micro_eur"] + amount > self.budget:
                raise QuotaExceededError("Monthly spending allowance exhausted")
            return {
                "used": state["used"] + 1,
                "spent_micro_eur": state["spent_micro_eur"] + amount,
            }

        self._update(change, month)
        return month

    def refund(self, month: str, amount: int) -> None:
        """Return unused micro-euros; on any failure the reservation stands."""
        if type(amount) is not int or amount <= 0:
            return
        try:
            self._update(
                lambda s: {
                    **s,
                    "spent_micro_eur": max(0, s["spent_micro_eur"] - amount),
                },
                month,
            )
        except (QuotaExceededError, QuotaUnavailableError):
            return


def _managed_identity_factory(client_id: str | None) -> Callable[[str], _Blob]:
    """Blob clients authenticated with the app's managed identity."""
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobClient

    credential = ManagedIdentityCredential(
        client_id=client_id or os.environ.get("AZURE_CLIENT_ID")
    )

    def build(url: str) -> _Blob:
        client: _Blob = BlobClient.from_blob_url(
            url,
            credential=credential,
            retry_total=1,
            connection_timeout=3,
            read_timeout=5,
        )
        return client

    return build


def quota_from_env() -> BlobQuota | None:
    """Build the allowance from ``QUOTA_CONTAINER_URL`` / ``QUOTA_MONTHLY_EUR``.

    Returns:
        The configured ledger, or ``None`` when ``QUOTA_CONTAINER_URL`` is unset.
    """
    url = os.environ.get("QUOTA_CONTAINER_URL")
    if not url:
        return None
    return BlobQuota(url, float(os.environ.get("QUOTA_MONTHLY_EUR", "1.0")))
