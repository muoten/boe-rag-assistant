"""Summarise what the corpus holds, for the frontend to state its own limits.

Counts documents and passages, looks up the first and last publication dates on
boe.es (the corpus has no date column), and estimates how much of the BOE's Section I
it covers from a seeded sample of days. Run it whenever the corpus changes; it needs
network access. The page reads the result at build time.

Example:
    python scripts/summarize_corpus.py \
        --corpus data/corpus/boe-2015-present.parquet --out reports/corpus_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

_BOE = "https://www.boe.es"
_DAYS_PER_YEAR = 5
_SEED = 1


def _get(url: str, accept: str) -> bytes:
    """Fetch a BOE URL."""
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "boe-rag-assistant"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        body: bytes = response.read()
    return body


def publication_date(document_id: str) -> date:
    """Return the date a document was published in the BOE."""
    xml = _get(f"{_BOE}/diario_boe/xml.php?id={document_id}", "application/xml")
    match = re.search(rb"<fecha_publicacion>(\d{8})", xml)
    if match is None:
        raise ValueError(f"No publication date for {document_id}")
    return datetime.strptime(match.group(1).decode(), "%Y%m%d").date()


def _identifiers(node: Any) -> Iterator[str]:
    """Yield every BOE document identifier nested in a sumario section."""
    if isinstance(node, dict):
        identifier = node.get("identificador")
        if isinstance(identifier, str) and identifier.startswith("BOE-A"):
            yield identifier
        for value in node.values():
            yield from _identifiers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _identifiers(value)


def section_one(day: date) -> list[str]:
    """Return the Section I documents of a day's issue; raises if there is none."""
    url = f"{_BOE}/datosabiertos/api/boe/sumario/{day:%Y%m%d}"
    sumario = json.loads(_get(url, "application/json"))["data"]["sumario"]
    return [
        identifier
        for section in _section_one_blocks(sumario["diario"])
        for identifier in _identifiers(section)
    ]


def _section_one_blocks(diarios: Any) -> Iterator[Any]:
    """Yield the Section I blocks of a sumario's issues."""
    for diario in diarios if isinstance(diarios, list) else [diarios]:
        sections = diario["seccion"]
        for section in sections if isinstance(sections, list) else [sections]:
            if section.get("codigo") == "1":
                yield section


def sample_days(first: date, last: date, seed: int = _SEED) -> list[date]:
    """Pick random days per calendar year between two dates, reproducibly."""
    rng = random.Random(seed)
    days = []
    for year in range(first.year, last.year + 1):
        start = max(first, date(year, 1, 1))
        span = (min(last, date(year, 12, 31)) - start).days
        days += [
            start + timedelta(days=rng.randrange(span + 1))
            for _ in range(_DAYS_PER_YEAR)
        ]
    return days


def _document_order(document_id: str) -> tuple[int, int]:
    """Sort key for ``BOE-A-<year>-<number>``, which numbers in publication order."""
    _, _, year, number = document_id.split("-")
    return int(year), int(number)


def summarize(
    corpus: Path,
    published: Callable[[str], date] = publication_date,
    issue: Callable[[date], list[str]] = section_one,
) -> dict[str, object]:
    """Summarise a corpus Parquet.

    Args:
        corpus: Corpus Parquet path.
        published: Looks up a document's publication date.
        issue: Lists a day's Section I documents; raises when there is no issue.

    Returns:
        The summary the page reads.
    """
    table = pq.read_table(corpus, columns=["document_id"])  # type: ignore[no-untyped-call]
    chunks = [str(value) for value in table.column("document_id").to_pylist()]
    documents = set(chunks)
    ordered = sorted(documents, key=_document_order)
    first, last = published(ordered[0]), published(ordered[-1])

    days = sampled = found = 0
    for day in sample_days(first, last):
        try:
            listed = issue(day)
        except (OSError, KeyError, ValueError):  # no BOE on that day
            continue
        days += 1
        sampled += len(listed)
        found += sum(identifier in documents for identifier in listed)
        time.sleep(0.35)
    return {
        "documents": len(documents),
        "passages": len(chunks),
        "first_published": first.isoformat(),
        "last_published": last.isoformat(),
        "section_one_sample": {"days": days, "documents": sampled, "in_corpus": found},
    }


def main(argv: list[str] | None = None) -> int:
    """Write the corpus summary.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Summarise the corpus.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus Parquet.")
    parser.add_argument(
        "--out", type=Path, default=Path("reports/corpus_summary.json"), help="Output."
    )
    args = parser.parse_args(argv)
    summary = summarize(args.corpus)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
