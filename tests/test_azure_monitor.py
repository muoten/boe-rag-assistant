"""Tests for the content-free latency telemetry, exporting to memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode

from boe_rag.llm.base import ChatMessage, LLMError
from boe_rag.service.api import create_app
from boe_rag.service.azure_monitor import (
    AzureMonitorTelemetry,
    RequestTelemetry,
)
from boe_rag.service.engine import ChunkInfo, RagEngine

_QUESTION = "¿Cuál es el tipo general del IVA?"
_ANSWER = "El tipo general es el 21% [1]."
_PASSAGE = "IVA general 21%."


class _Searcher:
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [("c1", 0.9)]


class _Provider:
    name = "fake"

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def complete(self, messages: Sequence[ChatMessage], **_: object) -> str:
        if self._fail:
            raise LLMError(f"provider said no to: {messages[-1].content}")
        return _ANSWER


def _telemetry(
    *, fail: bool = False
) -> tuple[TestClient, InMemorySpanExporter, AzureMonitorTelemetry]:
    exporter = InMemorySpanExporter()
    telemetry = AzureMonitorTelemetry(exporter)
    engine = RagEngine(
        retriever=_Searcher(),
        lookup={"c1": ChunkInfo(citation="Ley 1/2024", text=_PASSAGE, url="u1")},
        provider=_Provider(fail=fail),
        tracer=telemetry.tracer,
    )
    app = create_app(engine)
    app.add_middleware(RequestTelemetry, tracer=telemetry.otel_tracer)
    return TestClient(app), exporter, telemetry


def _spans(
    exporter: InMemorySpanExporter, telemetry: AzureMonitorTelemetry
) -> Mapping[str, ReadableSpan]:
    telemetry.provider.force_flush()
    return {span.name: span for span in exporter.get_finished_spans()}


def _exported_text(spans: Sequence[ReadableSpan]) -> str:
    return " ".join(span.to_json() for span in spans)


def test_ask_nests_every_stage_under_the_request() -> None:
    """Stage spans share the request's trace, so latency can be split by stage."""
    client, exporter, telemetry = _telemetry()

    assert client.post("/ask", json={"question": _QUESTION}).status_code == 200

    spans = _spans(exporter, telemetry)
    assert set(spans) == {"POST /ask", "boe.answer", "boe.retrieve", "boe.generate"}
    request = spans["POST /ask"]
    assert request.kind is SpanKind.SERVER
    assert request.attributes is not None
    assert request.attributes["http.response.status_code"] == 200
    assert request.attributes["http.route"] == "/ask"
    assert request.parent is None
    assert request.context is not None
    assert {s.context.trace_id for s in spans.values() if s.context} == {
        request.context.trace_id
    }
    answer = spans["boe.answer"]
    assert answer.parent is not None
    assert answer.parent.span_id == request.context.span_id
    assert answer.attributes is not None
    assert answer.attributes["boe.sources"] == 1


def test_no_question_or_answer_is_exported() -> None:
    """The exported spans hold no question, answer or passage text."""
    client, exporter, telemetry = _telemetry()

    client.post("/ask", json={"question": _QUESTION})
    client.post("/search", json={"query": _QUESTION})

    telemetry.provider.force_flush()
    exported = _exported_text(exporter.get_finished_spans())
    assert exported
    for content in ("IVA", "21%", "Ley 1/2024"):
        assert content not in exported


def test_a_failed_generation_exports_the_error_type_only() -> None:
    """A provider error marks the spans without exporting its message."""
    client, exporter, telemetry = _telemetry(fail=True)

    assert client.post("/ask", json={"question": _QUESTION}).status_code == 503

    spans = _spans(exporter, telemetry)
    generate = spans["boe.generate"]
    assert generate.status.status_code is StatusCode.ERROR
    assert generate.attributes is not None
    assert generate.attributes["error.type"] == "LLMError"
    assert spans["POST /ask"].status.status_code is StatusCode.ERROR
    assert not any(s.events for s in spans.values())
    assert "IVA" not in _exported_text(list(spans.values()))


def test_only_query_routes_are_traced_and_the_first_request_is_marked() -> None:
    """Probes aren't traced; only the first traced request counts as first."""
    client, exporter, telemetry = _telemetry()

    client.get("/health")
    client.post("/search", json={"query": _QUESTION})
    client.post("/search", json={"query": _QUESTION})

    telemetry.provider.force_flush()
    requests = [s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER]
    assert [s.name for s in requests] == ["POST /search", "POST /search"]
    flags = [s.attributes["boe.first_request"] for s in requests if s.attributes]
    assert flags == [True, False]


def test_record_boot_spans_the_engine_load() -> None:
    """The boot span starts when loading started, so it measures the wait."""
    exporter = InMemorySpanExporter()
    telemetry = AzureMonitorTelemetry(exporter)

    telemetry.record_boot(1_000)

    telemetry.provider.force_flush()
    boot = exporter.get_finished_spans()[0]
    assert boot.name == "boe.boot"
    assert boot.start_time == 1_000
