"""Tests for the metrics aggregated from saved answers and judge verdicts."""

from __future__ import annotations

import pytest

from boe_rag.eval.answerer import REFUSAL
from boe_rag.eval.standard_metrics import (
    JudgedAnswer,
    aggregate,
    count_words,
    extra_words,
    majority,
)


def _answered(
    answer: str = "El plazo es de 30 días [1].",
    *,
    correct: bool = True,
    grounded: bool = True,
    answerable: bool = True,
) -> JudgedAnswer:
    return JudgedAnswer(
        answer=answer,
        reference="30 días.",
        answerable=answerable,
        refused=False,
        correct=correct,
        grounded=grounded,
    )


def _refused(*, answerable: bool, correct: bool) -> JudgedAnswer:
    return JudgedAnswer(
        answer=REFUSAL,
        reference=REFUSAL if not answerable else "30 días.",
        answerable=answerable,
        refused=True,
        correct=correct,
        grounded=None,
    )


def test_citations_are_not_words() -> None:
    """Citation markers do not count, wherever they sit."""
    assert count_words("El plazo [1] es de 30 días [2].") == 6


def test_numbers_and_norm_references_are_one_word() -> None:
    """Figures and norm numbers count as a single word."""
    assert count_words("6.355.410.182 euros, 1,59 y Ley 39/2015.") == 6


def test_extra_words_never_go_below_zero() -> None:
    """An answer shorter than its reference has no extra words."""
    assert extra_words("El plazo es de 30 días.", "30 días.") == 4
    assert extra_words("30 días.", "El plazo es de 30 días.") == 0


def test_correctness_counts_refusals_against_every_question() -> None:
    """Refusals pass or fail like any answer, with plain error counts."""
    metrics = aggregate(
        [
            _answered(),
            _refused(answerable=True, correct=False),
            _refused(answerable=False, correct=True),
            _answered(correct=False, answerable=False),
        ]
    )
    assert metrics.correctness == 0.5
    assert metrics.wrong_refusals == 1
    assert metrics.wrong_answers == 1
    assert metrics.num_unanswerable == 2


def test_refusals_do_not_inflate_groundedness() -> None:
    """Groundedness only looks at answered questions."""
    metrics = aggregate(
        [
            _answered(grounded=False),
            _answered(),
            _refused(answerable=False, correct=True),
        ]
    )
    assert metrics.groundedness == 0.5


def test_always_refusing_has_no_groundedness_or_length() -> None:
    """A system that always refuses has nothing to measure there."""
    metrics = aggregate(
        [
            _refused(answerable=True, correct=False),
            _refused(answerable=False, correct=True),
        ]
    )
    assert metrics.groundedness is None
    assert metrics.median_extra_words is None
    assert metrics.correctness == 0.5


def test_extra_words_only_count_answers_passing_both_judges() -> None:
    """Failed answers and refusals do not affect the length."""
    long_answer = "El plazo es de 30 días, y el BOE se publica en Madrid."
    metrics = aggregate(
        [
            _answered("30 días y nada más que eso."),
            _answered(long_answer, grounded=False),
            _answered(long_answer, correct=False),
            _refused(answerable=False, correct=True),
        ]
    )
    assert metrics.median_extra_words == 5


def test_answered_question_needs_a_groundedness_verdict() -> None:
    """An answered question cannot skip the groundedness judge."""
    answer = JudgedAnswer("30 días.", "30 días.", True, False, True, None)
    with pytest.raises(ValueError, match="groundedness"):
        aggregate([answer])


def test_empty_run_is_rejected() -> None:
    """An empty run is an error, not a zero score."""
    with pytest.raises(ValueError, match="no answers"):
        aggregate([])


def test_majority_needs_an_odd_number_of_votes() -> None:
    """The majority of three is defined; a tie is rejected."""
    assert majority([True, False, True])
    assert not majority([True, False, False])
    with pytest.raises(ValueError, match="odd"):
        majority([True, False])


def test_non_unanimous_questions_are_counted() -> None:
    """Questions on which repeats disagreed are counted."""
    split = JudgedAnswer("30 días.", "30 días.", True, False, True, True, False)
    assert aggregate([split, _answered()]).non_unanimous == 1
