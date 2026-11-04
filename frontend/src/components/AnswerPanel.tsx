import type { AnswerResponse, Source } from "../types";
import { SourceCard } from "./SourceCard";

interface AnswerPanelProps {
  result: AnswerResponse;
}

/** Bracket numbers the answer cites, e.g. "… [2][3]." → {2, 3}.
 *
 * Mirrors the `[n]` pattern of the server's cite-or-refuse check
 * (src/boe_rag/service/citation.py), which already strips numbers that point
 * past the retrieved passages.
 */
export function citedIndices(answer: string): Set<number> {
  return new Set(Array.from(answer.matchAll(/\[(\d+)\]/g), (m) => Number(m[1])));
}

/** Renders a grounded answer with its cited sources, or a clear refusal notice.
 *
 * Only the passages the answer cites are listed as sources; the other retrieved
 * passages stay available, collapsed, under their own heading. Each keeps its
 * original number so the in-text `[n]` still points at the right card.
 */
export function AnswerPanel({ result }: AnswerPanelProps) {
  if (result.refused) {
    return (
      <section className="answer-panel refused" aria-live="polite">
        <p className="refusal">
          No he encontrado base suficiente en el BOE para responder a esa pregunta, así
          que prefiero no inventar. Prueba a reformularla o a preguntar por otra norma.
          Puede que la norma no esté en la parte del BOE consultada.
        </p>
      </section>
    );
  }

  const cited = citedIndices(result.answer);
  const numbered = result.sources.map((source, i) => ({ source, index: i + 1 }));
  const citedSources = numbered.filter(({ index }) => cited.has(index));
  const otherSources = numbered.filter(({ index }) => !cited.has(index));
  const renderList = (items: { source: Source; index: number }[]) => (
    <ol className="sources-list">
      {items.map(({ source, index }) => (
        <SourceCard key={source.chunk_id} source={source} index={index} />
      ))}
    </ol>
  );

  return (
    <section className="answer-panel" aria-live="polite">
      <p className="answer-text">{result.answer}</p>
      {citedSources.length > 0 && (
        <>
          <h2 className="sources-title">Fuentes citadas</h2>
          {renderList(citedSources)}
        </>
      )}
      {otherSources.length > 0 && (
        <details className="other-sources" open={citedSources.length === 0}>
          <summary className="sources-title">
            {citedSources.length > 0
              ? `Otros fragmentos recuperados (${otherSources.length})`
              : "Fragmentos recuperados"}
          </summary>
          {renderList(otherSources)}
        </details>
      )}
    </section>
  );
}
