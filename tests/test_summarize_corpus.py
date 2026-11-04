"""Tests for the corpus summary, with the BOE lookups faked."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _script() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "summarize_corpus.py"
    spec = importlib.util.spec_from_file_location("summarize_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_counts_documents_dates_and_sampled_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dates come from the numerically first and last IDs; empty days don't count."""
    module = _script()
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    corpus = tmp_path / "c.parquet"
    ids = ["BOE-A-2015-9", "BOE-A-2015-9", "BOE-A-2015-10", "BOE-A-2016-1"]
    pq.write_table(pa.table({"document_id": ids}), corpus)  # type: ignore[no-untyped-call]
    published = {
        "BOE-A-2015-9": date(2015, 1, 1),
        "BOE-A-2016-1": date(2016, 12, 31),
    }
    seen: list[date] = []

    def issue(day: date) -> list[str]:
        seen.append(day)
        if len(seen) % 2:
            raise KeyError("no issue")
        return ["BOE-A-2015-9", "BOE-A-2099-1"]

    summary = module.summarize(corpus, published.__getitem__, issue)

    assert summary["documents"] == 3
    assert summary["passages"] == 4
    assert summary["first_published"] == "2015-01-01"
    assert summary["last_published"] == "2016-12-31"
    assert len(seen) == 10
    assert summary["section_one_sample"] == {"days": 5, "documents": 10, "in_corpus": 5}


def test_sample_days_are_reproducible_and_in_range() -> None:
    """The same seed gives the same days, inside the corpus's dates."""
    module = _script()
    first, last = date(2015, 3, 1), date(2016, 2, 1)

    days = module.sample_days(first, last)

    assert days == module.sample_days(first, last)
    assert all(first <= day <= last for day in days)
    assert len(days) == 10
