"""Evaluate a running service with MLflow's built-in judges.

Two steps, so answers can be judged again without generating them again:

    # Spends generation budget on the service being asked.
    python scripts/run_standard_eval.py ask --api https://<service> \
        --evalset eval_data/standard_evalset.jsonl \
        --answers reports/standard_answers.jsonl

    # Spends judge budget. Needs the ``judges`` extra and the judge's credentials,
    # for a Foundry deployment OPENAI_API_BASE=<endpoint>/openai/v1 and an Entra ID
    # token in OPENAI_API_KEY.
    python scripts/run_standard_eval.py judge --judge-model openai:/<deployment> \
        --judge-label "gpt-6-luna 2026-09-22" \
        --evalset eval_data/standard_evalset.jsonl \
        --answers reports/standard_answers.jsonl \
        --out reports/standard_eval

``ask`` goes through ``POST /ask``, so it measures what is served: retrieval,
prompt and output guardrails together. ``judge`` calls ``is_correct`` and
``is_grounded``, the functions behind MLflow's ``Correctness`` and
``RetrievalGroundedness`` scorers, which spares recording a trace per answer.

A judge can disagree with itself on one input, so each verdict is the majority of
``--repeats`` calls. Both steps resume from what they have already saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from boe_rag.eval.answerer import REFUSAL
from boe_rag.eval.dataset import EvalExample, load_evalset
from boe_rag.eval.standard_metrics import (
    JudgedAnswer,
    StandardMetrics,
    aggregate,
    majority,
)

logger = logging.getLogger(__name__)

#: Seconds to wait for an answer, enough for a cold start of the service.
_ASK_TIMEOUT_S = 180.0

#: Characters per token assumed when pacing judge calls.
_CHARS_PER_TOKEN = 4


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load saved JSONL rows keyed by example id, or nothing if there is no file."""
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    return {row["example_id"]: row for row in rows}


def ask(api: str, examples: Sequence[EvalExample], answers_path: Path) -> int:
    """Ask the service every question not answered yet, saving each answer at once.

    Args:
        api: Base URL of the service.
        examples: The questions to ask.
        answers_path: JSONL file to resume from and append to.

    Returns:
        How many questions were asked.
    """
    done = _load_rows(answers_path)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [e for e in examples if e.example_id not in done]
    with answers_path.open("a", encoding="utf-8") as handle:
        for example in pending:
            response = httpx.post(
                f"{api.rstrip('/')}/ask",
                json={"question": example.question},
                timeout=_ASK_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
            row = {
                "example_id": example.example_id,
                "answer": body["answer"],
                "refused": body["refused"],
                "contexts": [
                    {"citation": s["citation"], "text": s["text"]}
                    for s in body["sources"]
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            logger.info("%s refused=%s", example.example_id, body["refused"])
    return len(pending)


def _expectation(example: EvalExample) -> dict[str, Any]:
    """Return what the correctness judge checks an answer against."""
    if not example.answerable:
        return {"expected_response": REFUSAL}
    if example.expected_facts:
        return {"expected_facts": list(example.expected_facts)}
    return {"expected_response": example.answer}


def _votes(
    call: Callable[[], Any], repeats: int, wait: float
) -> tuple[list[bool], list[str]]:
    """Call a judge repeatedly, failing rather than counting an error as a "no".

    Args:
        call: A judge call returning an MLflow ``Feedback``.
        repeats: How many times to call it.
        wait: Seconds to wait after each call, to stay under the rate limit.

    Returns:
        The verdicts and their rationales.

    Raises:
        RuntimeError: If the judge reports an error or an unknown verdict.
    """
    votes: list[bool] = []
    rationales: list[str] = []
    for _ in range(repeats):
        feedback = call()
        verdict = str(getattr(feedback.value, "value", feedback.value))
        if feedback.error or verdict not in ("yes", "no"):
            raise RuntimeError(f"judge failed: {feedback.error or verdict}")
        votes.append(verdict == "yes")
        rationales.append(feedback.rationale or "")
        time.sleep(wait)
    return votes, rationales


def _wait_seconds(text_chars: int, rpm: float, tpm: float) -> float:
    """Return the pause after a call of this size, for both rate limits."""
    return max(60.0 / rpm, text_chars / _CHARS_PER_TOKEN / tpm * 60.0)


def judge(
    examples: Sequence[EvalExample],
    saved: dict[str, dict[str, Any]],
    model: str,
    verdicts_path: Path,
    *,
    repeats: int = 3,
    rpm: float = 9.0,
    tpm: float = 9000.0,
) -> list[dict[str, Any]]:
    """Judge every saved answer with MLflow's built-in judges, resuming if stopped.

    Args:
        examples: The evaluated questions.
        saved: Saved answers keyed by example id.
        model: Judge model in MLflow's ``<provider>:/<model>`` form.
        verdicts_path: JSONL file to resume from and append to.
        repeats: Calls per verdict; odd, so the majority is defined.
        rpm: Requests per minute to stay under.
        tpm: Tokens per minute to stay under.

    Returns:
        One row per example, in example order, with votes and rationales.

    Raises:
        ValueError: If an example has no saved answer.
    """
    from mlflow.genai.judges import is_correct, is_grounded

    missing = [e.example_id for e in examples if e.example_id not in saved]
    if missing:
        raise ValueError(f"no saved answer for {missing}")
    done = _load_rows(verdicts_path)
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with verdicts_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            if example.example_id in done:
                continue
            row = saved[example.example_id]
            expectation = _expectation(example)
            correct, correct_why = _votes(
                partial(
                    is_correct,
                    request=example.question,
                    response=row["answer"],
                    model=model,
                    **expectation,
                ),
                repeats,
                _wait_seconds(
                    len(example.question) + len(row["answer"]) + 1500, rpm, tpm
                ),
            )
            grounded: list[bool] | None = None
            grounded_why: list[str] | None = None
            if not row["refused"]:
                grounded, grounded_why = _votes(
                    partial(
                        is_grounded,
                        request=example.question,
                        response=row["answer"],
                        context=row["contexts"],
                        model=model,
                    ),
                    repeats,
                    _wait_seconds(
                        len(row["answer"]) + len(str(row["contexts"])) + 1500, rpm, tpm
                    ),
                )
            judged = {
                **row,
                "correct_votes": correct,
                "correct_rationales": correct_why,
                "grounded_votes": grounded,
                "grounded_rationales": grounded_why,
            }
            done[example.example_id] = judged
            handle.write(json.dumps(judged, ensure_ascii=False) + "\n")
            handle.flush()
            logger.info(
                "%s correct=%s grounded=%s", example.example_id, correct, grounded
            )
    return [done[e.example_id] for e in examples]


def _judged_answers(
    examples: Sequence[EvalExample], rows: Sequence[dict[str, Any]]
) -> list[JudgedAnswer]:
    """Join judged rows with their examples, in example order."""
    by_id = {row["example_id"]: row for row in rows}
    answers = []
    for example in examples:
        row = by_id[example.example_id]
        grounded = row["grounded_votes"]
        answers.append(
            JudgedAnswer(
                answer=row["answer"],
                reference=REFUSAL if not example.answerable else (example.answer or ""),
                answerable=example.answerable,
                refused=row["refused"],
                correct=majority(row["correct_votes"]),
                grounded=None if grounded is None else majority(grounded),
                unanimous=all(
                    len(set(votes)) == 1
                    for votes in (row["correct_votes"], grounded or [True])
                ),
            )
        )
    return answers


def _render_report(metrics: StandardMetrics, judge_label: str, repeats: int) -> str:
    """Render the metrics as a Markdown report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            "# End-to-end evaluation with MLflow's judges",
            "",
            f"- **Generated:** {timestamp}",
            f"- **Judge:** `{judge_label}`, majority of {repeats} calls per verdict",
            f"- **Questions:** {metrics.num_questions}, "
            f"of which {metrics.num_unanswerable} unanswerable",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Correctness | {metrics.correctness:.3f} |",
            f"| Groundedness (answered only) | {metrics.groundedness} |",
            f"| Median extra words | {metrics.median_extra_words} |",
            f"| Wrong refusals | {metrics.wrong_refusals} |",
            f"| Wrong answers | {metrics.wrong_answers} |",
            f"| Questions with non-unanimous verdicts | {metrics.non_unanimous} |",
            "",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    for name in ("ask", "judge"):
        step = steps.add_parser(name)
        step.add_argument("--evalset", type=Path, required=True)
        step.add_argument("--answers", type=Path, required=True)
        if name == "ask":
            step.add_argument("--api", required=True, help="Service base URL.")
        else:
            step.add_argument("--judge-model", required=True)
            step.add_argument("--judge-label", required=True, help="Model and version.")
            step.add_argument("--out", type=Path, required=True, help="Output stem.")
            step.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one step of the evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    examples = load_evalset(args.evalset)
    if args.step == "ask":
        asked = ask(args.api, examples, args.answers)
        logger.info("Asked %d questions; answers in %s", asked, args.answers)
        return 0
    rows = judge(
        examples,
        _load_rows(args.answers),
        args.judge_model,
        args.out.with_suffix(".verdicts.jsonl"),
        repeats=args.repeats,
    )
    metrics = aggregate(_judged_answers(examples, rows))
    args.out.with_suffix(".md").write_text(
        _render_report(metrics, args.judge_label, args.repeats), encoding="utf-8"
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "judge": args.judge_label,
                "repeats": args.repeats,
                **metrics.as_dict(),
                "answers": rows,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote report to %s.{md,json}", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
