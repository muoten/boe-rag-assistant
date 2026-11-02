"""Tests for weighted RRF and the fusion-ablation helpers."""

from __future__ import annotations

import pytest

from boe_rag.eval.equivalence import build_text_equivalence
from boe_rag.eval.fusion import (
    has_legal_reference,
    lexical_overlap,
    reachable,
    score_ranking,
)
from boe_rag.eval.hybrid import reciprocal_rank_fusion

_BM25 = ["A", "B", "C", "D"]
_DENSE = ["C", "E", "A", "F"]


def test_agreement_outranks_a_single_high_rank() -> None:
    """With k=60, chunks both legs return beat a chunk only one ranks second."""
    fused = [cid for cid, _ in reciprocal_rank_fusion([_BM25, _DENSE], 60)]
    assert set(fused[:2]) == {"A", "C"}
    assert fused.index("B") > fused.index("C")


def test_small_k_lets_a_single_first_place_win() -> None:
    """With k=1, first place in one list beats fifth place in both."""
    one = ["X", "p", "q", "r", "Y"]
    other = ["s", "t", "u", "v", "Y"]
    fused = [cid for cid, _ in reciprocal_rank_fusion([one, other], 1)]
    assert fused.index("X") < fused.index("Y")
    fused = [cid for cid, _ in reciprocal_rank_fusion([one, other], 60)]
    assert fused.index("Y") < fused.index("X")


def test_weights_scale_each_rankings_contribution() -> None:
    """Weighting a leg twice breaks the A/C tie in its favour."""
    fused = reciprocal_rank_fusion([_BM25, _DENSE], 60, weights=[2.0, 1.0])
    scores = dict(fused)
    assert fused[0][0] == "A"
    assert scores["A"] == pytest.approx(2 / 61 + 1 / 63)
    assert reciprocal_rank_fusion([_BM25, _DENSE], 60, [1.0, 1.0]) == (
        reciprocal_rank_fusion([_BM25, _DENSE], 60)
    )


def test_weights_must_match_the_rankings() -> None:
    """A weight per ranking is required."""
    with pytest.raises(ValueError, match="one weight per ranking"):
        reciprocal_rank_fusion([_BM25, _DENSE], 60, weights=[1.0])


def test_score_ranking_reports_rank_and_cutoffs() -> None:
    """Gold at rank 3: hit at 5 and 10, reciprocal rank one third."""
    scores = score_ranking(["x", "y", "gold", "z"], {"gold"})
    assert scores["first_rank"] == 3
    assert scores["recall@5"] == scores["recall@10"] == 1.0
    assert scores["mrr"] == pytest.approx(1 / 3)
    missing = score_ranking(["x", "y"], {"gold"})
    assert (missing["first_rank"], missing["recall@5"], missing["mrr"]) == (0, 0, 0)


def test_identical_copies_of_the_gold_chunk_count() -> None:
    """Text equivalence credits a byte-identical duplicate and collapses ranks."""
    equivalence = build_text_equivalence(
        ["gold", "copy", "dup1", "dup2"], ["same", "same", "twice", "twice"]
    )
    scores = score_ranking(["dup1", "dup2", "copy"], {"gold"}, equivalence)
    assert scores["first_rank"] == 2
    assert score_ranking(["copy"], {"gold"})["first_rank"] == 0


def test_reachable_checks_every_leg_to_the_depth() -> None:
    """The ceiling holds if any leg has the gold chunk within the depth."""
    assert reachable([_BM25, _DENSE], {"E"}, depth=2)
    assert not reachable([_BM25, _DENSE], {"F"}, depth=3)
    assert reachable([_BM25, _DENSE], {"F"}, depth=4)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("¿Qué regula la Ley 39/2015?", True),
        ("¿Qué dice el artículo 14 sobre plazos?", True),
        ("¿Qué establece el Real Decreto 1133/2023?", True),
        ("¿Qué aprueba la Orden HAC/1495/2025?", True),
        ("¿Dónde tiene su sede el Instituto Vasco de Finanzas?", False),
        ("¿Qué empresa asume el voto por correo en 2024?", False),
    ],
)
def test_legal_references(question: str, expected: bool) -> None:
    """Numbered norms and articles are references; bare years are not."""
    assert has_legal_reference(question) is expected


def test_lexical_overlap_counts_shared_content_terms() -> None:
    """Stopwords are ignored; the share is over distinct question terms."""
    overlap = lexical_overlap("¿Cuál es el plazo de recurso?", "El plazo es breve")
    assert overlap == pytest.approx(1 / 3)  # cuál, plazo, recurso; only plazo shared
    assert lexical_overlap("¿Y el de la?", "cualquier texto") == 0.0
