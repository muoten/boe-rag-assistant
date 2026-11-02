"""Helpers for the fusion ablation: per-query scoring and query slicing.

The ablation (``scripts/run_fusion_ablation.py``) retrieves each leg's ranking once
per question and derives every fused variant from those cached rankings. These
helpers score one ranking against its gold chunks, report whether the gold chunk
was reachable at all, and tag questions so results can be sliced by *why* a
retriever wins: an explicit legal reference (BM25's strength) and how much of the
question's wording the gold passage shares (which flatters BM25 on LLM-generated
questions written from that passage).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from boe_rag.eval.equivalence import TextEquivalence
from boe_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from boe_rag.eval.sparse import tokenize_es

#: Cut-offs reported per query: 5 is what ``/ask`` grounds an answer in, 10 is
#: what ``/search`` shows.
CUTOFFS = (5, 10)

#: An explicit reference to a numbered norm or provision.
_LEGAL_REFERENCE = re.compile(
    r"\b(?:ley|real decreto(?:-ley)?|decreto(?:-ley)?|orden|resoluci[oó]n|reglamento)"
    r"\s+(?:org[aá]nica\s+)?(?:[A-Z]{2,4}/)?\d+/\d{4}"
    r"|\bart[ií]culos?\s+\d+"
    r"|\b\d+/\d{4}\b",
    re.IGNORECASE,
)


def has_legal_reference(question: str) -> bool:
    """Whether the question names a numbered norm (``Ley 39/2015``) or article."""
    return _LEGAL_REFERENCE.search(question) is not None


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's distinct content terms that occur in the passage.

    Uses the BM25 tokenizer, so stopwords and one-character tokens are ignored.
    Returns 0.0 for a question with no content terms.
    """
    terms = set(tokenize_es(question))
    if not terms:
        return 0.0
    return len(terms & set(tokenize_es(passage))) / len(terms)


def reachable(
    rankings: Sequence[Sequence[str]],
    relevant: Collection[str],
    depth: int,
    equivalence: TextEquivalence | None = None,
) -> bool:
    """Whether any gold chunk appears in the top ``depth`` of any ranking.

    This is the ceiling for fusion and reranking over those candidates: a chunk
    neither leg retrieves cannot be recovered downstream.
    """
    gold = _canonical_set(relevant, equivalence)
    for ranking in rankings:
        if gold & _canonical_set(ranking[:depth], equivalence):
            return True
    return False


def score_ranking(
    ranking: Sequence[str],
    relevant: Collection[str],
    equivalence: TextEquivalence | None = None,
) -> dict[str, float]:
    """Per-query metrics for one ranking, with optional text equivalence.

    Returns ``first_rank`` (1-based rank of the first gold chunk, 0 if absent),
    ``recall@k`` for each of :data:`CUTOFFS`, ``mrr`` and ``ndcg@10``. With an
    equivalence, byte-identical copies of a gold chunk count as hits and
    duplicates collapse to one rank, as in ``boe_rag.eval.runner``.
    """
    if equivalence is not None:
        ranked: Sequence[str] = equivalence.canonical_sequence(ranking)
        gold: Collection[str] = equivalence.canonical_set(relevant)
    else:
        ranked, gold = ranking, set(relevant)
    first = next((i for i, cid in enumerate(ranked, start=1) if cid in gold), 0)
    scores = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in CUTOFFS}
    scores["mrr"] = reciprocal_rank(ranked, gold)
    scores["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    scores["first_rank"] = float(first)
    return scores


def _canonical_set(
    ids: Collection[str] | Sequence[str], equivalence: TextEquivalence | None
) -> frozenset[str]:
    """Ids mapped to their canonical class (unchanged without an equivalence)."""
    return equivalence.canonical_set(ids) if equivalence else frozenset(ids)
