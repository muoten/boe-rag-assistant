import { type FormEvent, useState } from "react";
import EXAMPLES from "../examples.json";

interface AskFormProps {
  onAsk: (question: string) => void;
  loading: boolean;
  placeholder?: string;
  label?: string;
  loadingLabel?: string;
}

// Every example must be answerable from the served corpus (BOE Disposiciones
// Generales, 2015–present). The first two come from the gold eval set (scored
// 1.00/1.00 end-to-end); the third passes every judge in reports/standard_eval.md.
// Changing them needs scripts/pregenerate-examples.mjs, or they wait for the service.

/** The query input with a submit button and a few example prompts.
 *
 * The button/placeholder copy is configurable so the same form serves both the
 * "ask" (grounded answer) and "search" (raw retrieval) modes.
 */
export function AskForm({
  onAsk,
  loading,
  placeholder = "Pregunta sobre legislación española…",
  label = "Preguntar",
  loadingLabel = "Consultando…",
}: AskFormProps) {
  const [value, setValue] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !loading) {
      onAsk(trimmed);
    }
  }

  return (
    <form className="ask-form" onSubmit={submit}>
      <div className="ask-row">
        <input
          className="ask-input"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          maxLength={1000}
          disabled={loading}
        />
        <button
          className="ask-button"
          type="submit"
          disabled={loading || !value.trim()}
        >
          {loading ? loadingLabel : label}
        </button>
      </div>
      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="example-chip"
            onClick={() => {
              setValue(example);
              onAsk(example);
            }}
            disabled={loading}
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
