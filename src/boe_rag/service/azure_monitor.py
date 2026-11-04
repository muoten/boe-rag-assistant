"""Content-free latency telemetry for Azure Application Insights.

Enabled with ``BOE_TELEMETRY=azure_monitor`` (needs the ``azure`` extra). Requests
to ``/ask`` and ``/search`` and each pipeline stage become spans, so latency can be
split into retrieval and generation. Spans carry durations, counts and flags only:
questions, answers, URLs and exception messages never leave the process.

Ingestion authenticates with the managed identity in ``AZURE_CLIENT_ID``;
``APPLICATIONINSIGHTS_CONNECTION_STRING`` only names the destination.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from boe_rag.service.tracing import Span

#: The Application Insights role name, shared workspace with the rag-poc app.
SERVICE_NAME = "boe-rag"
#: Only these requests are traced; probes and preflights would just use up the cap.
_TRACED_ROUTES = ("/ask", "/search")


def _content_free(fields: Mapping[str, object]) -> dict[str, bool | int | float]:
    """Keep numbers and flags: anything textual could hold a user's question."""
    return {
        f"boe.{key}": value
        for key, value in fields.items()
        if isinstance(value, bool | int | float)
    }


class _AzureSpan:
    """Adapts an OpenTelemetry span to the tracing ``Span`` protocol."""

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def update(
        self,
        *,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Record numeric and boolean metadata; ``output`` is the answer, so drop it."""
        if metadata is not None:
            self._span.set_attributes(_content_free(metadata))


class AzureMonitorTracer:
    """A ``Tracer`` that exports stage timings, never their inputs or outputs."""

    def __init__(self, tracer: trace.Tracer) -> None:
        """Bind the OpenTelemetry tracer spans are opened on."""
        self._tracer = tracer

    @contextmanager
    def span(self, name: str, **inputs: object) -> Iterator[Span]:
        """Open a span for a pipeline stage, keeping only content-free inputs."""
        with self._tracer.start_as_current_span(
            f"boe.{name}",
            attributes=_content_free(inputs),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield _AzureSpan(span)
            except Exception as err:
                # The type only: a provider's message may quote the prompt.
                span.set_attribute("error.type", type(err).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise


class RequestTelemetry:
    """ASGI middleware opening the request span that stage spans nest under.

    Marks the first request of each process, which pays for whatever was left to
    warm up. The wait for the container to boot happens before the app sees the
    request, so it isn't in this span: see ``boe.boot``.
    """

    def __init__(self, app: ASGIApp, tracer: trace.Tracer) -> None:
        """Wrap ``app``, opening request spans on ``tracer``."""
        self._app = app
        self._tracer = tracer
        self._first_request = True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Trace ``POST`` requests to the query routes; pass the rest through."""
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in _TRACED_ROUTES
        ):
            await self._app(scope, receive, send)
            return
        first_request, self._first_request = self._first_request, False
        with self._tracer.start_as_current_span(
            f"POST {scope['path']}",
            context=Context(),  # clients can't choose our trace identity
            kind=SpanKind.SERVER,
            attributes={
                "http.request.method": "POST",
                "http.route": scope["path"],
                "boe.first_request": first_request,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:

            async def traced_send(message: Message) -> None:
                if message["type"] == "http.response.start":
                    span.set_attribute("http.response.status_code", message["status"])
                    if message["status"] >= 500:
                        span.set_status(Status(StatusCode.ERROR))
                await send(message)

            try:
                await self._app(scope, receive, traced_send)
            except Exception as err:
                span.set_attribute("error.type", type(err).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise


class AzureMonitorTelemetry:
    """The tracer provider behind the stage tracer and the request middleware."""

    def __init__(self, exporter: SpanExporter) -> None:
        """Export spans in the background through ``exporter``."""
        # A private provider: no global state and no automatic HTTP or SDK capture.
        # It flushes on interpreter exit, so a scale-down keeps its last spans.
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": SERVICE_NAME})
        )
        self.provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=256,
                max_export_batch_size=32,
                schedule_delay_millis=1000,
            )
        )
        self.otel_tracer = self.provider.get_tracer("boe_rag", "1")
        self.tracer = AzureMonitorTracer(self.otel_tracer)

    def record_boot(self, started_ns: int) -> None:
        """Record how long loading the engine took, the bulk of a cold start."""
        self.otel_tracer.start_span(
            "boe.boot", context=Context(), start_time=started_ns
        ).end()


@lru_cache(maxsize=1)
def azure_monitor_telemetry() -> AzureMonitorTelemetry:
    """Build the process-wide telemetry that exports to Application Insights."""
    # SDK self-reporting isn't ours to send, and would use up the workspace cap.
    os.environ["APPLICATIONINSIGHTS_STATSBEAT_DISABLED_ALL"] = "true"
    os.environ["APPLICATIONINSIGHTS_SDKSTATS_DISABLED"] = "true"
    from azure.identity import ManagedIdentityCredential
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

    return AzureMonitorTelemetry(
        AzureMonitorTraceExporter(
            connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
            credential=ManagedIdentityCredential(
                client_id=os.environ["AZURE_CLIENT_ID"]
            ),
            disable_offline_storage=True,
            timeout=5,
            read_timeout=5,
        )
    )
