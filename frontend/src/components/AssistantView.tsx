import { useEffect, useState } from "react";
import {
  ApiError,
  PREGENERATED_ON,
  ask,
  pregeneratedAnswer,
  pregeneratedResults,
  search,
} from "../api";
import type { AnswerResponse, SearchResponse } from "../types";
import { AnswerPanel } from "./AnswerPanel";
import { AskForm } from "./AskForm";
import { SearchResults } from "./SearchResults";

type Status = "idle" | "loading" | "done" | "error";
type Mode = "ask" | "search";

const LOADING_COPY: Record<Mode, string> = {
  ask: "Buscando en el BOE y redactando una respuesta fundamentada…",
  search: "Recuperando los pasajes más relevantes del BOE…",
};

/** The loading notice: what's happening, what to expect, and the time so far.
 *
 * The service scales to zero, so the first request after a while waits about a
 * minute with nothing else to show for it.
 */
function LoadingStatus({ mode }: { mode: Mode }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(
      () => setSeconds(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="status loading">
      <span className="spinner" aria-hidden="true" />
      <div aria-live="polite">
        {LOADING_COPY[mode]}
        <p className="status-hint">
          Suele tardar unos segundos. Si el servicio estaba en reposo, puede tardar
          alrededor de un minuto en arrancar.{" "}
          <span aria-hidden="true" className="status-elapsed">
            {seconds} s
          </span>
        </p>
      </div>
    </div>
  );
}

/** The interactive tool: ask for a grounded answer, or search raw passages. */
export function AssistantView() {
  const [mode, setMode] = useState<Mode>("ask");
  const [status, setStatus] = useState<Status>("idle");
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  function switchMode(next: Mode) {
    if (next === mode) {
      return;
    }
    setMode(next);
    setStatus("idle");
    setAnswer(null);
    setResults(null);
    setError(null);
    setAsked(null);
  }

  async function handleSubmit(query: string) {
    setStatus("loading");
    setError(null);
    setAnswer(null);
    setResults(null);
    setAsked(query);
    try {
      if (mode === "ask") {
        const savedAnswer = pregeneratedAnswer(query);
        setSaved(savedAnswer !== undefined);
        setAnswer(savedAnswer ?? (await ask(query)));
      } else {
        const savedResults = pregeneratedResults(query);
        setSaved(savedResults !== undefined);
        setResults(savedResults ?? (await search(query)));
      }
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Ha ocurrido un error inesperado.",
      );
      setStatus("error");
    }
  }

  return (
    <>
      <div className="mode-toggle" role="tablist" aria-label="Modo de consulta">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "ask"}
          className={`mode-tab ${mode === "ask" ? "active" : ""}`}
          onClick={() => switchMode("ask")}
        >
          Preguntar
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "search"}
          className={`mode-tab ${mode === "search" ? "active" : ""}`}
          onClick={() => switchMode("search")}
        >
          Buscar pasajes
        </button>
      </div>

      <AskForm
        onAsk={handleSubmit}
        loading={status === "loading"}
        placeholder={
          mode === "ask"
            ? "Pregunta sobre legislación española…"
            : "Busca términos en la legislación española…"
        }
        label={mode === "ask" ? "Preguntar" : "Buscar"}
      />

      {asked && status !== "idle" && (
        <p className="asked-question">
          <span className="asked-label">
            {mode === "ask" ? "Pregunta:" : "Búsqueda:"}
          </span>{" "}
          {asked}
        </p>
      )}

      {status === "loading" && <LoadingStatus mode={mode} />}

      {status === "error" && error && (
        <div className="status error" role="alert">
          {error}
        </div>
      )}

      {status === "done" && saved && (
        <p className="status-hint pregenerated-note">
          {mode === "ask" ? "Respuesta pregenerada" : "Resultados pregenerados"} el{" "}
          {PREGENERATED_ON}. Lo que escribas tú se consulta en el momento.
        </p>
      )}

      {status === "done" && mode === "ask" && answer && <AnswerPanel result={answer} />}
      {status === "done" && mode === "search" && results && (
        <SearchResults result={results} />
      )}
    </>
  );
}
