r"""Grow a corpus with the documents of randomly chosen days, to test a larger one.

Used to measure what a fuller corpus does to retrieval, latency and memory without
crawling all of it. Whole days are sampled, seeded, across the corpus's date span,
and the existing passages keep their order, so results stay comparable. ``crawl`` is
resumable; ``embed`` then embeds only the new passages as "title. text" and appends
them to the existing matrix.

Example:
    python scripts/extend_corpus_sample.py crawl \
        --corpus data/corpus/boe-2015-present.parquet \
        --factor 2 --out data/corpus/boe-pilot-x2.parquet
    python scripts/extend_corpus_sample.py embed \
        --base-embeddings data/corpus/boe-embeddings-title.npz \
        --corpus data/corpus/boe-pilot-x2.parquet --out data/corpus/boe-pilot-x2.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from boe_rag.config import IngestionConfig
from boe_rag.ingest.client import BoeClient
from boe_rag.ingest.corpus import CORPUS_SCHEMA, write_corpus
from boe_rag.ingest.pipeline import ingest_dates
from boe_rag.models import Chunk

logger = logging.getLogger(__name__)


def shuffled_days(first: date, last: date, seed: int) -> list[date]:
    """Return every day in the span in a reproducible random order."""
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    random.Random(seed).shuffle(days)
    return days


def crawl(args: argparse.Namespace) -> int:
    """Add sampled days until the corpus reaches ``factor`` times its passages."""
    base = pq.read_table(args.corpus)  # type: ignore[no-untyped-call]
    known = {str(value) for value in base.column("document_id").to_pylist()}
    target = (args.factor - 1) * base.num_rows
    work = args.out.with_suffix(".jsonl")
    done: set[str] = set()
    added = 0
    if work.is_file():  # resume: each line is one finished day
        with work.open(encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                done.add(record["day"])
                added += len(record["chunks"])

    config = IngestionConfig()
    with BoeClient(config) as client, work.open("a", encoding="utf-8") as out:
        for day in shuffled_days(args.first, args.last, args.seed):
            if added >= target:
                break
            if day.isoformat() in done:
                continue
            chunks = [
                chunk
                for chunk in ingest_dates([day], config, client)
                if chunk.document_id not in known
            ]
            record = {
                "day": day.isoformat(),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            added += len(chunks)
            logger.info("%s: +%d passages (%d/%d)", day, len(chunks), added, target)

    new: list[Chunk] = []
    days: list[str] = []
    with work.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            days.append(record["day"])
            new.extend(Chunk.model_validate(chunk) for chunk in record["chunks"])
    # Resuming can repeat the day it stopped on.
    unique = list({chunk.chunk_id: chunk for chunk in new}.values())
    extra = args.out.with_suffix(".new.parquet")
    write_corpus(unique, extra)
    merged = pa.concat_tables(
        [
            base.select(CORPUS_SCHEMA.names).cast(CORPUS_SCHEMA),
            pq.read_table(extra).select(CORPUS_SCHEMA.names).cast(CORPUS_SCHEMA),  # type: ignore[no-untyped-call]
        ]
    )
    pq.write_table(merged, args.out)  # type: ignore[no-untyped-call]
    extra.unlink()
    summary = {
        "seed": args.seed,
        "days_sampled": len(days),
        "new_passages": len(unique),
        "new_documents": len({chunk.document_id for chunk in unique}),
        "passages": merged.num_rows,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def embed(args: argparse.Namespace) -> int:
    """Embed the passages the base corpus lacks and append them to its matrix."""
    from boe_rag.eval.embedding import DEFAULT_MODEL, E5Embedder
    from boe_rag.eval.retriever import load_embeddings, save_embeddings

    base_ids, base_matrix = load_embeddings(args.base_embeddings)
    corpus = pq.read_table(
        args.corpus, columns=["chunk_id", "document_title", "text"]
    ).to_pydict()  # type: ignore[no-untyped-call]
    ids = [str(value) for value in corpus["chunk_id"]]
    if ids[: len(base_ids)] != base_ids:
        raise SystemExit("The corpus does not start with the base embeddings' passages")

    embedder = E5Embedder(DEFAULT_MODEL, device="cpu")
    parts = [base_matrix]
    start = len(base_ids)
    for i in range(start, len(ids), args.batch):
        texts = [
            f"{title}. {text}"
            for title, text in zip(
                corpus["document_title"][i : i + args.batch],
                corpus["text"][i : i + args.batch],
                strict=True,
            )
        ]
        parts.append(embedder.embed_passages(texts))
        logger.info("Embedded %d/%d", i + len(texts), len(ids))
    matrix = np.concatenate(parts)
    save_embeddings(args.out, ids, matrix)
    print(f"Saved {matrix.shape} to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a step.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    step = steps.add_parser("crawl")
    step.add_argument("--corpus", type=Path, required=True)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--factor", type=float, default=2.0)
    step.add_argument("--seed", type=int, default=0)
    step.add_argument("--first", type=date.fromisoformat, default=date(2015, 1, 1))
    step.add_argument("--last", type=date.fromisoformat, default=date(2026, 9, 14))
    step.set_defaults(run=crawl)
    step = steps.add_parser("embed")
    step.add_argument("--base-embeddings", type=Path, required=True)
    step.add_argument("--corpus", type=Path, required=True)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--batch", type=int, default=512)
    step.set_defaults(run=embed)
    args = parser.parse_args(argv)
    status: int = args.run(args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
