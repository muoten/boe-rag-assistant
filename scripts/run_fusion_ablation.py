"""Fusion ablation: BM25 vs dense vs RRF variants, with and without reranking.

Answers "does hybrid retrieval beat BM25 alone, and is RRF configured well?" on
the production corpus, with paired significance tests instead of bare means.

Each question's BM25 and dense rankings are retrieved **once** (top ``--depth``)
and cached; every fused variant is derived from those cached rankings, so the
RRF constant, the per-leg candidate depth and leg weights can be compared at no
extra retrieval cost. The served configuration (50 candidates per leg, k_rrf=10,
no reranking: ``BOE_RRF_K=10``, ``BOE_RERANK=false``) is always included, also
with the fused top 30 reranked by the cross-encoder.

Question sets, reported separately:

- **gold** — the hand-curated golden set, never used to choose anything;
- **norm-refs** — hand-checked "¿Qué regula <norma>?" questions whose gold chunk
  is the norm's own object article: the query is little more than a norm number,
  which is where BM25 tokenisation of ``N/YYYY`` matters;
- **silver-dev** / **silver-test** — the generated silver set split by source
  document (``eval.split``). The best fused variant is *chosen* on dev and
  *reported* on test, so tuning does not flatter the result.

Per set and system: recall@5 (what ``/ask`` grounds an answer in), recall@10 (what
``/search`` shows), MRR and nDCG@10, each with a 95% bootstrap CI; paired sign-flip
tests for the key pairs; the candidate ceiling (gold reachable by either leg);
and silver-test slices by explicit legal reference and question/passage word
overlap. Scoring credits byte-identical copies of a gold chunk, as the committed
baseline does.

Requires the ``ml`` extra. The silver set is ``generated_evalset.jsonl`` from the
``gonzalonao/boe-rag-evalset`` Hub dataset.

Example:
    python scripts/run_fusion_ablation.py \
        --corpus data/corpus/boe-2015-present.parquet \
        --embeddings data/corpus/boe-2015-present-embeddings.npz \
        --silver data/evalset/generated_evalset.jsonl \
        --out reports/retrieval_fusion
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from boe_rag.eval.dataset import EvalExample, load_evalset
from boe_rag.eval.equivalence import TextEquivalence, build_text_equivalence
from boe_rag.eval.fusion import (
    has_legal_reference,
    lexical_overlap,
    reachable,
    score_ranking,
)
from boe_rag.eval.hybrid import reciprocal_rank_fusion
from boe_rag.eval.split import split_by_document
from boe_rag.eval.stats import bootstrap_mean_ci, paired_delta_significance

logger = logging.getLogger(__name__)

#: Metrics reported per system (keys of ``score_ranking``).
METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")
#: Metrics the paired tests and slices use.
TESTED = ("recall@5", "mrr")
SERVED = "rrf k10 d50 (served)"
#: What upstream serves: k_rrf=60, then the cross-encoder rerank.
UPSTREAM = "rrf k60 d50 (upstream)"


@dataclass(frozen=True, slots=True)
class Fusion:
    """An RRF configuration over the cached legs."""

    depth: int
    k_rrf: int
    weights: tuple[float, float] = (1.0, 1.0)  # (dense, bm25)

    def apply(self, dense: Sequence[str], bm25: Sequence[str]) -> list[str]:
        """Fuse the top ``depth`` of each leg."""
        fused = reciprocal_rank_fusion(
            [dense[: self.depth], bm25[: self.depth]], self.k_rrf, self.weights
        )
        return [cid for cid, _ in fused]


#: Fused variants. Depth 20 is the setting of the committed hybrid ablation.
FUSIONS: dict[str, Fusion] = {
    SERVED: Fusion(50, 10),
    UPSTREAM: Fusion(50, 60),
    "rrf k60 d20 (ablation)": Fusion(20, 60),
    "rrf k60 d100": Fusion(100, 60),
    "rrf k30 d50": Fusion(50, 30),
    "rrf k100 d50": Fusion(50, 100),
    "rrf k60 d50 bm25x2": Fusion(50, 60, (1.0, 2.0)),
    "rrf k60 d50 densex2": Fusion(50, 60, (2.0, 1.0)),
}


def _load_corpus(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load chunk ids, texts and the texts the BM25 leg indexes (with titles)."""
    from boe_rag.eval.sparse import BM25_COLUMNS, bm25_texts

    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "text", *BM25_COLUMNS]
    )
    data = table.to_pydict()
    texts = list(map(str, data["text"]))
    sparse_texts = bm25_texts(
        list(map(str, data["document_id"])),
        list(map(str, data["document_title"])),
        list(map(int, data["ordinal"])),
        data["articulo"],
        texts,
    )
    return list(map(str, data["chunk_id"])), texts, sparse_texts


def _in_corpus(
    examples: list[EvalExample], ids: set[str]
) -> tuple[list[EvalExample], int]:
    """Keep examples with at least one gold chunk in the corpus."""
    kept = [e for e in examples if any(c in ids for c in e.relevant_chunk_ids)]
    return kept, len(examples) - len(kept)


def _retrieve_legs(
    examples: Sequence[EvalExample],
    chunk_ids: list[str],
    sparse_texts: list[str],
    embeddings: Path,
    depth: int,
    cache: Path,
) -> dict[str, dict[str, list[str]]]:
    """Top-``depth`` BM25 and dense rankings per example, cached on disk."""
    from boe_rag.eval.sparse import tokenize_es

    # The BM25 leg depends on the tokenizer and on what it indexes; a probe
    # string and the first indexed text key the cache to both.
    stamp = {
        "chunks": len(chunk_ids),
        "depth": depth,
        "embeddings": embeddings.name,
        "tokens": tokenize_es("Ley 39/2015, artículo 14"),
        "bm25_text": sparse_texts[0][:200],
    }
    legs: dict[str, dict[str, list[str]]] = {}
    if cache.is_file():
        stored = json.loads(cache.read_text())
        if stored.get("stamp") == stamp:
            legs = stored["legs"]
    todo = [e for e in examples if e.example_id not in legs]
    if not todo:
        return legs

    from boe_rag.eval.embedding import E5Embedder
    from boe_rag.eval.retriever import DenseRetriever, load_embeddings
    from boe_rag.eval.sparse import BM25Index

    cached_ids, matrix = load_embeddings(embeddings)
    if cached_ids != chunk_ids:
        raise SystemExit("Embeddings do not match the corpus order; re-fetch both.")
    dense = DenseRetriever(E5Embedder())
    dense.index_precomputed(chunk_ids, matrix)
    logger.info("Indexing BM25 over %d chunks ...", len(chunk_ids))
    bm25 = BM25Index()
    bm25.index(chunk_ids, sparse_texts)
    logger.info("Retrieving %d questions from both legs ...", len(todo))
    start = time.monotonic()
    for n, example in enumerate(todo, start=1):
        legs[example.example_id] = {
            "dense": [cid for cid, _ in dense.search(example.question, depth)],
            "bm25": [cid for cid, _ in bm25.search(example.question, depth)],
        }
        if n % 200 == 0 or n == len(todo):
            logger.info("  %d/%d (%.0fs)", n, len(todo), time.monotonic() - start)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"stamp": stamp, "legs": legs}))
    return legs


def _rerank_all(
    examples: Sequence[EvalExample],
    rankings: dict[str, list[str]],
    text_of: dict[str, str],
    pool: int,
) -> dict[str, list[str]]:
    """Cross-encoder reorder of each example's top ``pool`` candidates."""
    from boe_rag.eval.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    out: dict[str, list[str]] = {}
    start = time.monotonic()
    for n, example in enumerate(examples, start=1):
        candidates = [(c, text_of[c]) for c in rankings[example.example_id][:pool]]
        ranked = reranker.rerank(example.question, candidates)
        out[example.example_id] = [cid for cid, _ in ranked]
        if n % 200 == 0 or n == len(examples):
            logger.info(
                "  reranked %d/%d (%.0fs)", n, len(examples), time.monotonic() - start
            )
    return out


def _summarise(
    per_query: dict[str, dict[str, dict[str, float]]], ids: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and 95% bootstrap CI per system and metric over ``ids``."""
    return {
        system: {
            m: bootstrap_mean_ci([scores[i][m] for i in ids]).as_dict() for m in METRICS
        }
        for system, scores in per_query.items()
        if all(i in scores for i in ids)
    }


def _paired(
    per_query: dict[str, dict[str, dict[str, float]]],
    ids: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    """Paired sign-flip tests, candidate minus baseline, for each pair."""
    rows: list[dict[str, object]] = []
    for baseline, candidate in pairs:
        if baseline not in per_query or candidate not in per_query:
            continue
        for metric in TESTED:
            sig = paired_delta_significance(
                [per_query[baseline][i][metric] for i in ids],
                [per_query[candidate][i][metric] for i in ids],
            )
            row: dict[str, object] = {"baseline": baseline, "candidate": candidate}
            row |= {"metric": metric, **sig.as_dict()}
            rows.append(row)
    return rows


def _slices(
    per_query: dict[str, dict[str, dict[str, float]]],
    ids: Sequence[str],
    tags: dict[str, dict[str, object]],
    systems: Sequence[str],
) -> list[dict[str, object]]:
    """Recall@5 and MRR per system within question slices."""
    overlaps = np.array([float(str(tags[i]["overlap"])) for i in ids])
    low, high = np.quantile(overlaps, [1 / 3, 2 / 3]) if len(ids) else (0.0, 0.0)

    def bucket(i: str) -> str:
        value = float(str(tags[i]["overlap"]))
        return "low" if value <= low else "high" if value > high else "mid"

    groups: dict[str, Callable[[str], bool]] = {
        "legal reference": lambda i: bool(tags[i]["legal_reference"]),
        "no legal reference": lambda i: not tags[i]["legal_reference"],
        f"overlap low (<= {low:.2f})": lambda i: bucket(i) == "low",
        "overlap mid": lambda i: bucket(i) == "mid",
        f"overlap high (> {high:.2f})": lambda i: bucket(i) == "high",
    }
    rows: list[dict[str, object]] = []
    for name, keep in groups.items():
        members = [i for i in ids if keep(i)]
        row: dict[str, object] = {"slice": name, "n": len(members)}
        for system in systems:
            if system in per_query and members:
                for metric in TESTED:
                    values = [per_query[system][i][metric] for i in members]
                    row[f"{system} | {metric}"] = sum(values) / len(values)
        rows.append(row)
    return rows


def _render(results: dict[str, object]) -> str:
    """Markdown report."""
    meta = results["meta"]
    assert isinstance(meta, dict)
    lines = [
        "# Fusion ablation — BM25 vs dense vs RRF (+ rerank)",
        "",
        *(f"- **{k}:** {v}" for k, v in meta.items()),
        "",
        "Recall@5 is what `/ask` grounds an answer in; recall@10 is what `/search`",
        "shows. Values are means with 95% bootstrap intervals. *Reachable@d* is the",
        "share of questions whose gold chunk is in either leg's top d — the ceiling",
        "for any fusion or reranking over those candidates.",
    ]
    sets = results["sets"]
    assert isinstance(sets, dict)
    for name, block in sets.items():
        summary, ceiling = block["summary"], block["ceiling"]
        lines += ["", f"## {name} ({block['n']} questions)", ""]
        lines.append(
            "Reachable: " + ", ".join(f"@{d} {v:.3f}" for d, v in ceiling.items())
        )
        lines += ["", "| System | " + " | ".join(METRICS) + " |"]
        lines.append("|---" * (len(METRICS) + 1) + "|")
        for system, metrics in summary.items():
            cells = [
                f"{m['point']:.3f} [{m['low']:.2f}, {m['high']:.2f}]"
                for m in (metrics[k] for k in METRICS)
            ]
            lines.append(f"| {system} | " + " | ".join(cells) + " |")
        if block.get("paired"):
            lines += [
                "",
                "| Candidate vs baseline | Metric | Δ | 95% CI | p |",
                "|---|---|---|---|---|",
            ]
            for row in block["paired"]:
                lines.append(
                    f"| {row['candidate']} vs {row['baseline']} | {row['metric']} | "
                    f"{row['delta']:+.3f} | [{row['low']:+.3f}, {row['high']:+.3f}] | "
                    f"{row['p_value']:.3f} |"
                )
        if block.get("slices"):
            systems = sorted(
                {k.split(" | ")[0] for r in block["slices"] for k in r if " | " in k}
            )
            lines += [
                "",
                "| Slice | n | " + " | ".join(f"{s} R@5 / MRR" for s in systems) + " |",
            ]
            lines.append("|---|---" + "|---" * len(systems) + "|")
            for row in block["slices"]:
                cells = [
                    f"{row.get(f'{s} | recall@5', float('nan')):.3f} / "
                    f"{row.get(f'{s} | mrr', float('nan')):.3f}"
                    for s in systems
                ]
                lines.append(
                    f"| {row['slice']} | {row['n']} | " + " | ".join(cells) + " |"
                )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the fusion ablation.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus .parquet.")
    parser.add_argument(
        "--embeddings", type=Path, required=True, help="Precomputed E5 .npz."
    )
    parser.add_argument(
        "--gold", type=Path, default=Path("eval_data/seed_evalset.jsonl")
    )
    parser.add_argument(
        "--norm-refs",
        type=Path,
        default=Path("eval_data/norm_reference_evalset.jsonl"),
        help="Norm-reference questions; skipped if the file is missing.",
    )
    parser.add_argument("--silver", type=Path, help="Generated evalset .jsonl.")
    parser.add_argument("--limit-silver", type=int, help="Use the first N silver.")
    parser.add_argument("--depth", type=int, default=100, help="Cached per leg.")
    parser.add_argument("--rerank-pool", type=int, default=30)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/fusion_legs.json"),
        help="Cached leg rankings (reused when corpus and depth match).",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/retrieval_fusion"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ablation and write the .md, .json and per-question .csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    chunk_ids, texts, sparse_texts = _load_corpus(args.corpus)
    id_set, text_of = set(chunk_ids), dict(zip(chunk_ids, texts, strict=True))
    equivalence: TextEquivalence = build_text_equivalence(chunk_ids, texts)

    gold, gold_dropped = _in_corpus(load_evalset(args.gold), id_set)
    sets: dict[str, list[EvalExample]] = {"gold": gold}
    if args.norm_refs.is_file():
        sets["norm-refs"], _ = _in_corpus(load_evalset(args.norm_refs), id_set)
    silver_dropped = 0
    if args.silver:
        silver = load_evalset(args.silver)[: args.limit_silver]
        silver, silver_dropped = _in_corpus(silver, id_set)
        dev, test = split_by_document(
            silver, test_fraction=args.test_fraction, seed=args.seed
        )
        sets |= {"silver-dev": dev, "silver-test": test}
    examples = [e for group in sets.values() for e in group]
    by_id = {e.example_id: e for e in examples}

    legs = _retrieve_legs(
        examples, chunk_ids, sparse_texts, args.embeddings, args.depth, args.cache
    )
    rankings: dict[str, dict[str, list[str]]] = {
        "bm25": {i: legs[i]["bm25"] for i in by_id},
        "dense": {i: legs[i]["dense"] for i in by_id},
    }
    for name, fusion in FUSIONS.items():
        rankings[name] = {
            i: fusion.apply(legs[i]["dense"], legs[i]["bm25"]) for i in by_id
        }

    def score_all(system: str) -> dict[str, dict[str, float]]:
        return {
            i: score_ranking(r, by_id[i].relevant_chunk_ids, equivalence)
            for i, r in rankings[system].items()
        }

    per_query = {system: score_all(system) for system in rankings}

    # Choose the best fused variant on silver-dev (or gold if no silver) by MRR.
    select_on = "silver-dev" if "silver-dev" in sets else "gold"
    select_ids = [e.example_id for e in sets[select_on]]
    best = max(
        FUSIONS,
        key=lambda s: (
            np.mean([per_query[s][i]["mrr"] for i in select_ids]),
            np.mean([per_query[s][i]["recall@5"] for i in select_ids]),
        ),
    )
    logger.info("Best fused variant on %s: %s", select_on, best)

    if not args.no_rerank:
        report_on = [
            e for name, group in sets.items() if name != "silver-dev" for e in group
        ]
        for base in dict.fromkeys([SERVED, UPSTREAM, best]):
            logger.info("Reranking %s for %d questions ...", base, len(report_on))
            reranked = _rerank_all(report_on, rankings[base], text_of, args.rerank_pool)
            name = f"{base} + rerank"
            rankings[name] = reranked
            per_query[name] = score_all(name)

    tags: dict[str, dict[str, object]] = {
        i: {
            "legal_reference": has_legal_reference(e.question),
            "overlap": round(
                lexical_overlap(
                    e.question,
                    next(text_of[c] for c in e.relevant_chunk_ids if c in text_of),
                ),
                3,
            ),
        }
        for i, e in by_id.items()
    }
    pairs = [
        ("bm25", "dense"),
        ("bm25", SERVED),
        ("dense", SERVED),
        (SERVED, UPSTREAM),
        (SERVED, f"{UPSTREAM} + rerank"),
        (SERVED, "rrf k60 d20 (ablation)"),
        (SERVED, best),
        (SERVED, f"{SERVED} + rerank"),
        (f"{SERVED} + rerank", f"{best} + rerank"),
    ]
    pairs = [(a, b) for a, b in dict.fromkeys(pairs) if a != b]
    blocks: dict[str, dict[str, object]] = {}
    for name, group in sets.items():
        ids = [e.example_id for e in group]
        if not ids:
            continue
        block: dict[str, object] = {
            "n": len(ids),
            "ceiling": {
                d: float(
                    np.mean(
                        [
                            reachable(
                                [legs[i]["dense"], legs[i]["bm25"]],
                                by_id[i].relevant_chunk_ids,
                                d,
                                equivalence,
                            )
                            for i in ids
                        ]
                    )
                )
                for d in (20, 50, 100)
            },
            "summary": _summarise(per_query, ids),
        }
        if name != "silver-dev":
            block["paired"] = _paired(per_query, ids, pairs)
            block["slices"] = _slices(
                per_query,
                ids,
                tags,
                ["bm25", "dense", SERVED, f"{SERVED} + rerank"],
            )
        blocks[name] = block

    results: dict[str, object] = {
        "meta": {
            "Generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "Corpus": f"`{args.corpus.name}` ({len(chunk_ids)} chunks, "
            f"{equivalence.num_redundant} byte-identical duplicates folded)",
            "Cached depth per leg": args.depth,
            "Served": "50 per leg, k_rrf=10, no rerank (BOE_RRF_K, BOE_RERANK)",
            "BM25 tokens": "norm numbers kept whole (`39/2015` is one token)",
            "BM25 text": "document title + text; title twice on head chunks",
            "Best fused variant": f"{best} (chosen on {select_on} by MRR)",
            "Dropped (gold not in corpus)": (
                f"gold {gold_dropped}, silver {silver_dropped}"
            ),
            "Silver split": (
                f"by document, test {args.test_fraction}, seed {args.seed}"
            ),
        },
        "sets": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    args.out.with_suffix(".md").write_text(_render(results), encoding="utf-8")
    systems = list(per_query)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set",
                "example_id",
                "legal_reference",
                "overlap",
                "question",
                *(f"rank: {s}" for s in systems),
            ]
        )
        for name, group in sets.items():
            for e in group:
                writer.writerow(
                    [
                        name,
                        e.example_id,
                        tags[e.example_id]["legal_reference"],
                        tags[e.example_id]["overlap"],
                        e.question,
                        *(
                            int(per_query[s][e.example_id]["first_rank"])
                            if e.example_id in per_query[s]
                            else ""
                            for s in systems
                        ),
                    ]
                )
    logger.info("Wrote %s.{md,json,csv}", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
