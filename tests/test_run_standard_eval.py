"""Tests for the MLflow evaluation script, without a service or a judge."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from boe_rag.eval.answerer import REFUSAL
from boe_rag.eval.dataset import EvalExample, load_evalset

_REPO = Path(__file__).parents[1]
_EVALSET = _REPO / "eval_data" / "standard_evalset.jsonl"


def _load_script() -> ModuleType:
    """Import the script by path, since ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "run_standard_eval", _REPO / "scripts" / "run_standard_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()

_ANSWERABLE = EvalExample(
    example_id="a",
    question="¿Qué plazo hay?",
    relevant_chunk_ids=("c1",),
    answer="30 días.",
    expected_facts=("El plazo es de 30 días",),
)
_UNANSWERABLE = EvalExample(example_id="u", question="¿Y el IVA?", answerable=False)


class _Response:
    """A successful ``/ask`` response."""

    def __init__(self, question: str) -> None:
        self.question = question

    def raise_for_status(self) -> None:
        """Never fails."""

    def json(self) -> dict[str, Any]:
        """Return a served answer with one source."""
        return {
            "answer": f"Sobre {self.question} [1]",
            "refused": False,
            "sources": [{"citation": "Ley 1", "text": "Texto", "url": "u", "score": 1}],
        }


def test_unanswerable_example_needs_no_chunks_and_answerable_does() -> None:
    """An example has relevant chunks exactly when it is answerable."""
    assert _UNANSWERABLE.relevant_chunk_ids == ()
    with pytest.raises(ValueError, match="relevant chunks"):
        EvalExample(
            example_id="x", question="¿?", relevant_chunk_ids=("c1",), answerable=False
        )


def test_expectation_prefers_facts_and_expects_the_refusal() -> None:
    """Answerable questions check facts; unanswerable ones expect the refusal."""
    assert script._expectation(_ANSWERABLE) == {
        "expected_facts": ["El plazo es de 30 días"]
    }
    assert script._expectation(_UNANSWERABLE) == {"expected_response": REFUSAL}


def test_ask_saves_answers_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run asks only what the answers file does not hold yet."""
    asked: list[str] = []

    def post(url: str, json: dict[str, str], timeout: float) -> _Response:
        assert url == "https://api.test/ask"
        asked.append(json["question"])
        return _Response(json["question"])

    monkeypatch.setattr(script.httpx, "post", post)
    answers = tmp_path / "answers.jsonl"

    assert script.ask("https://api.test/", [_ANSWERABLE], answers) == 1
    assert script.ask("https://api.test/", [_ANSWERABLE, _UNANSWERABLE], answers) == 1

    assert asked == [_ANSWERABLE.question, _UNANSWERABLE.question]
    saved = [json.loads(line) for line in answers.read_text("utf-8").splitlines()]
    assert [row["example_id"] for row in saved] == ["a", "u"]
    assert saved[0]["contexts"] == [{"citation": "Ley 1", "text": "Texto"}]


def test_judged_answers_use_the_refusal_as_unanswerable_reference() -> None:
    """Rows join with their examples; split votes are flagged, not hidden."""
    row = {
        "example_id": "u",
        "answer": REFUSAL,
        "refused": True,
        "correct_votes": [True, False, True],
        "grounded_votes": None,
    }
    judged = script._judged_answers([_UNANSWERABLE], [row])
    assert judged[0].reference == REFUSAL
    assert not judged[0].answerable
    assert judged[0].correct
    assert not judged[0].unanimous


def _feedback(value: str, error: str | None = None) -> SimpleNamespace:
    """Build a stand-in for MLflow's ``Feedback``."""
    return SimpleNamespace(value=value, rationale="porque", error=error)


def _fake_judges(
    monkeypatch: pytest.MonkeyPatch, correct: list[Any], grounded: list[Any]
) -> list[str]:
    """Install fake MLflow judges returning these feedbacks; record the calls."""
    calls: list[str] = []

    def is_correct(**kwargs: Any) -> Any:
        calls.append("correct")
        return correct.pop(0)

    def is_grounded(**kwargs: Any) -> Any:
        calls.append("grounded")
        return grounded.pop(0)

    judges = ModuleType("mlflow.genai.judges")
    judges.is_correct = is_correct  # type: ignore[attr-defined]
    judges.is_grounded = is_grounded  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.genai", ModuleType("mlflow.genai"))
    monkeypatch.setitem(sys.modules, "mlflow.genai.judges", judges)
    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    return calls


_SAVED = {
    "a": {"example_id": "a", "answer": "30 días [1].", "refused": False, "contexts": []}
}


def test_judge_takes_repeated_votes_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each verdict is three calls; a second run repeats nothing already saved."""
    calls = _fake_judges(
        monkeypatch,
        [_feedback("yes"), _feedback("no"), _feedback("yes")],
        [_feedback("yes"), _feedback("yes"), _feedback("yes")],
    )
    path = tmp_path / "verdicts.jsonl"

    rows = script.judge([_ANSWERABLE], _SAVED, "openai:/x", path)
    again = script.judge([_ANSWERABLE], _SAVED, "openai:/x", path)

    assert calls == ["correct"] * 3 + ["grounded"] * 3
    assert rows[0]["correct_votes"] == [True, False, True]
    assert again[0]["grounded_votes"] == [True, True, True]


def test_judge_error_stops_the_run_instead_of_counting_as_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed judge call must not become a "no" verdict."""
    _fake_judges(monkeypatch, [_feedback("no", error="rate limited")], [])
    path = tmp_path / "verdicts.jsonl"
    with pytest.raises(RuntimeError, match="rate limited"):
        script.judge([_ANSWERABLE], _SAVED, "openai:/x", path)
    assert path.read_text("utf-8") == ""


def test_pause_respects_both_rate_limits() -> None:
    """Small calls are paced by requests, large ones by tokens."""
    assert script._wait_seconds(400, rpm=10, tpm=10000) == 6.0
    assert script._wait_seconds(8000, rpm=10, tpm=10000) == 12.0


def test_standard_evalset_is_consistent() -> None:
    """Answerable examples carry facts and a reference; ids are unique."""
    examples = load_evalset(_EVALSET)
    assert len({e.example_id for e in examples}) == len(examples)
    answerable = [e for e in examples if e.answerable]
    assert all(e.expected_facts and e.answer for e in answerable)
    assert all(not e.expected_facts for e in examples if not e.answerable)
    assert sum(not e.answerable for e in examples) == 10
