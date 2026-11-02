"""Tests for the blob-backed spending allowance, against an in-memory blob."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("azure.core")

from azure.core.exceptions import (
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)

from boe_rag.llm.quota import (
    BlobQuota,
    QuotaExceededError,
    QuotaUnavailableError,
    micro,
)

_URL = "https://example.blob.core.windows.net/quota"


class _Download:
    def __init__(self, data: str, etag: int) -> None:
        self._data = data
        self.properties = type("Props", (), {"etag": etag})()

    def readall(self) -> str:
        return self._data


class _Blob:
    """One blob with ETags; ``conflicts`` simulates concurrent writers."""

    def __init__(self, data: str | None = None, conflicts: int = 0) -> None:
        self.data, self.etag, self.conflicts = data, 0, conflicts

    def download_blob(self) -> _Download:
        if self.data is None:
            raise ResourceNotFoundError("missing")
        return _Download(self.data, self.etag)

    def upload_blob(self, data: str, **kwargs: Any) -> None:
        if not kwargs.get("overwrite"):
            if self.data is not None:
                raise ResourceExistsError("exists")
        elif self.conflicts:
            self.conflicts -= 1
            self.etag += 1
            raise ResourceModifiedError("changed")
        elif kwargs.get("etag") != self.etag:
            raise ResourceModifiedError("stale")
        self.data, self.etag = data, self.etag + 1


def _state(blob: _Blob) -> dict[str, int]:
    assert blob.data is not None
    loaded: dict[str, int] = json.loads(blob.data)
    return loaded


def test_reserve_creates_the_month_and_stops_at_the_budget() -> None:
    """Reservations accumulate until the next one would pass the budget."""
    blob = _Blob()
    quota = BlobQuota(_URL, 0.003, factory=lambda _: blob)
    quota.reserve(2000)
    with pytest.raises(QuotaExceededError):
        quota.reserve(1001)
    quota.reserve(1000)
    assert _state(blob) == {"used": 2, "spent_micro_eur": micro(0.003)}


def test_refund_returns_unused_spend() -> None:
    """A refund lowers the month's spend but never below zero."""
    blob = _Blob()
    quota = BlobQuota(_URL, 1, factory=lambda _: blob)
    month = quota.reserve(2000)
    quota.refund(month, 1500)
    assert _state(blob)["spent_micro_eur"] == 500
    quota.refund(month, 9999)
    assert _state(blob)["spent_micro_eur"] == 0


def test_concurrent_writes_are_retried() -> None:
    """An ETag conflict re-reads and retries instead of overwriting."""
    blob = _Blob(json.dumps({"used": 0, "spent_micro_eur": 0}), conflicts=2)
    BlobQuota(_URL, 1, factory=lambda _: blob).reserve(10)
    assert _state(blob)["spent_micro_eur"] == 10


def test_corrupt_or_unreachable_state_is_never_reset() -> None:
    """Unknown state refuses admission instead of starting from zero."""
    blob = _Blob(json.dumps({"used": -1, "spent_micro_eur": 0}))
    with pytest.raises(QuotaUnavailableError):
        BlobQuota(_URL, 1, factory=lambda _: blob).reserve(1)

    def broken(_: str) -> _Blob:
        raise OSError("network down")

    with pytest.raises(QuotaUnavailableError):
        BlobQuota(_URL, 1, factory=broken).reserve(1)


def test_failed_refund_leaves_the_reservation() -> None:
    """Refunds are best effort; a failure keeps the higher spend."""
    blob = _Blob()
    quota = BlobQuota(_URL, 1, factory=lambda _: blob)
    month = quota.reserve(100)
    blob.conflicts = 99
    quota.refund(month, 100)
    assert _state(blob)["spent_micro_eur"] == 100


@pytest.mark.parametrize(
    "url",
    [
        "http://example.blob.core.windows.net/quota",
        "https://example.com/quota",
        "https://example.blob.core.windows.net/quota/nested",
        "https://example.blob.core.windows.net/quota?sig=secret",
    ],
)
def test_rejects_unexpected_container_urls(url: str) -> None:
    """Only a credential-free container URL is accepted."""
    with pytest.raises(ValueError):
        BlobQuota(url, 1, factory=lambda _: _Blob())


def test_budget_must_be_small_and_positive() -> None:
    """The allowance is bounded to a demo-sized range."""
    for budget in (0, 11):
        with pytest.raises(ValueError):
            BlobQuota(_URL, budget, factory=lambda _: _Blob())
