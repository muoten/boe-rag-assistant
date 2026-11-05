import pregenerated from "./pregenerated-examples.json";
import type { AnswerResponse, SearchResponse, Source } from "./types";

// Base URL of the JSON API. Injected at build time (VITE_API_BASE_URL = the
// deployed HF Space); empty string means same-origin (useful for local proxying).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/** An error carrying a user-facing, Spanish message ready to render. */
export class ApiError extends Error {}

const MESSAGES: Record<number, string> = {
  429: "Demasiadas peticiones seguidas. Espera unos segundos e inténtalo de nuevo.",
  503: "El servicio de generación no está disponible ahora mismo. Vuelve a intentarlo en un momento.",
};

/** A cold start took about a minute when measured; this is well past that. */
const TIMEOUT_MS = 120_000;

/** POST a JSON body to an API endpoint, mapping failures to a typed ApiError. */
async function postJson<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new ApiError(
        "El servicio no ha respondido en dos minutos. Inténtalo de nuevo; si estaba arrancando, ahora debería ir más rápido.",
      );
    }
    throw new ApiError(
      "No se ha podido conectar con el servicio. Comprueba tu conexión e inténtalo de nuevo.",
    );
  }
  if (!response.ok) {
    throw new ApiError(
      MESSAGES[response.status] ?? `Error inesperado (${response.status}).`,
    );
  }
  return (await response.json()) as T;
}

/** The day the example responses were saved by scripts/pregenerate-examples.mjs. */
export const PREGENERATED_ON = new Date(pregenerated.generated).toLocaleDateString(
  "es-ES",
  { dateStyle: "long", timeZone: "UTC" },
);

// Maps, since a typed question such as "constructor" would find a plain object's own.
const SAVED_ANSWERS = new Map<string, AnswerResponse>(Object.entries(pregenerated.ask));
const SAVED_RESULTS = new Map<string, SearchResponse>(
  Object.entries(pregenerated.search),
);

/** The saved answer to exactly this question, which needs no running service. */
export function pregeneratedAnswer(question: string): AnswerResponse | undefined {
  return SAVED_ANSWERS.get(question);
}

/** The saved passages for exactly this query, which need no running service. */
export function pregeneratedResults(query: string): SearchResponse | undefined {
  return SAVED_RESULTS.get(query);
}

/**
 * Ask the assistant a question and get a grounded, cited answer.
 *
 * @param question Natural-language question about Spanish legislation.
 * @param k Number of passages to ground the answer in (1–20).
 * @throws ApiError with a user-facing message on a non-2xx response or network failure.
 */
export function ask(question: string, k = 5): Promise<AnswerResponse> {
  return postJson<AnswerResponse>("/ask", { question, k });
}

/**
 * Retrieve the raw passages ranked for a query, without generating an answer.
 *
 * @param query The search query about Spanish legislation.
 * @param k Number of passages to return (1–20).
 * @throws ApiError with a user-facing message on a non-2xx response or network failure.
 */
export function search(query: string, k = 10): Promise<SearchResponse> {
  return postJson<SearchResponse>("/search", { query, k });
}

/**
 * The passages closest to a refused question, or none if they cannot be fetched.
 *
 * Failing here must not hide the refusal, which is already a complete response.
 *
 * @param query The refused question.
 * @param k Number of passages to return.
 */
export async function closestPassages(query: string, k = 5): Promise<Source[]> {
  try {
    return (await search(query, k)).results;
  } catch {
    return [];
  }
}
