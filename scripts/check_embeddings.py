"""Fail unless a precomputed embedding matrix fits a corpus, chunk for chunk.

The service would otherwise re-encode the whole corpus at boot, without titles, so
the Azure image build runs this after baking in its matrix.

Example:
    python scripts/check_embeddings.py \
        --corpus data/corpus/boe-2015-present.parquet \
        --embeddings data/corpus/boe-embeddings-title.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]


def main(argv: list[str] | None = None) -> int:
    """Compare the chunk ids of the corpus and the matrix.

    Returns:
        Process exit code (0 when they match).
    """
    parser = argparse.ArgumentParser(description="Check embeddings against a corpus.")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Corpus Parquet path."
    )
    parser.add_argument("--embeddings", type=Path, required=True, help="The .npz path.")
    args = parser.parse_args(argv)

    table = pq.read_table(args.corpus, columns=["chunk_id"])  # type: ignore[no-untyped-call]
    corpus_ids = [str(cid) for cid in table.column("chunk_id").to_pylist()]
    with np.load(args.embeddings, allow_pickle=True) as data:
        matrix_ids = [str(cid) for cid in data["chunk_ids"]]
        rows = len(data["matrix"])
    if matrix_ids != corpus_ids or rows != len(corpus_ids):
        print(
            f"ERROR: {args.embeddings} ({rows} rows) does not match "
            f"{args.corpus} ({len(corpus_ids)} chunks)",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {rows} embeddings match the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
