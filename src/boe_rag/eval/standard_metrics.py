"""Aggregate end-to-end metrics from saved answers and judge verdicts.

The set stays small on purpose. Each answer is judged yes/no with MLflow's built-in
judges (see ``scripts/run_standard_eval.py``), and this module adds the parts that
need no LLM:

* **correctness**: pass rate over every question. An unanswerable question expects
  the refusal, so wrong refusals and wrong answers both fail it.
* **groundedness**: pass rate over answered questions only, since a refusal states
  nothing that a passage could support.
* **extra words**: median words above the reference answer, over answers that pass
  both judges, so failed answers and refusals cannot improve it.

Wrong refusals and wrong answers are plain counts next to the score, not metrics.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

#: Citation markers such as ``[1]``, which are not part of the answer's wording.
_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")

#: A word; "1,59", "6.355.410.182" and "39/2015" each count as one.
_WORD_RE = re.compile(r"\w+(?:[.,/]\w+)*")


def count_words(text: str) -> int:
    """Count the words of a text, ignoring citation markers and punctuation."""
    return len(_WORD_RE.findall(_CITATION_RE.sub(" ", text)))


def extra_words(answer: str, reference: str) -> int:
    """Words the answer has above its reference answer, with a minimum of 0."""
    return max(0, count_words(answer) - count_words(reference))


@dataclass(frozen=True, slots=True)
class JudgedAnswer:
    """One saved answer with its verdicts.

    Attributes:
        answer: The served answer.
        reference: The reference answer.
        answerable: Whether the corpus holds the answer.
        refused: Whether the service refused, as it reported.
        correct: The correctness verdict.
        grounded: The groundedness verdict, or ``None`` for a refusal.
        unanimous: Whether every repeat of the judges agreed.
    """

    answer: str
    reference: str
    answerable: bool
    refused: bool
    correct: bool
    grounded: bool | None
    unanimous: bool = True


@dataclass(frozen=True, slots=True)
class StandardMetrics:
    """Aggregated metrics of one run.

    Attributes:
        num_questions: Questions evaluated.
        num_unanswerable: How many of them the corpus cannot answer.
        correctness: Pass rate over every question.
        groundedness: Pass rate over answered questions, ``None`` if none answered.
        median_extra_words: Median extra words over answers passing both judges,
            ``None`` if none passed.
        wrong_refusals: Answerable questions that were refused.
        wrong_answers: Unanswerable questions that were answered.
        non_unanimous: Questions on which repeats of a judge disagreed.
    """

    num_questions: int
    num_unanswerable: int
    correctness: float
    groundedness: float | None
    median_extra_words: float | None
    wrong_refusals: int
    wrong_answers: int
    non_unanimous: int

    def as_dict(self) -> dict[str, float | int | None]:
        """Return the metrics as a plain dict for serialisation."""
        return asdict(self)


def aggregate(answers: Sequence[JudgedAnswer]) -> StandardMetrics:
    """Aggregate judged answers into :class:`StandardMetrics`.

    Args:
        answers: The judged answers of one run.

    Returns:
        The aggregated metrics.

    Raises:
        ValueError: If ``answers`` is empty, or an answered question has no
            groundedness verdict.
    """
    if not answers:
        raise ValueError("no answers to aggregate")
    answered = [a for a in answers if not a.refused]
    if any(a.grounded is None for a in answered):
        raise ValueError("every answered question needs a groundedness verdict")
    passing = [a for a in answered if a.answerable and a.correct and a.grounded]
    return StandardMetrics(
        num_questions=len(answers),
        num_unanswerable=sum(not a.answerable for a in answers),
        correctness=sum(a.correct for a in answers) / len(answers),
        groundedness=(
            sum(bool(a.grounded) for a in answered) / len(answered)
            if answered
            else None
        ),
        median_extra_words=(
            statistics.median(extra_words(a.answer, a.reference) for a in passing)
            if passing
            else None
        ),
        wrong_refusals=sum(a.answerable and a.refused for a in answers),
        wrong_answers=sum(not a.answerable and not a.refused for a in answers),
        non_unanimous=sum(not a.unanimous for a in answers),
    )


def majority(votes: Sequence[bool]) -> bool:
    """Return the majority of an odd number of verdicts.

    Args:
        votes: The repeated verdicts on one answer.

    Returns:
        The verdict most repeats gave.

    Raises:
        ValueError: If the number of votes is even, so a tie is possible.
    """
    if len(votes) % 2 == 0:
        raise ValueError("majority needs an odd number of votes")
    return sum(votes) * 2 > len(votes)
