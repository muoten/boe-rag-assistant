// Save the live service's responses to the example questions.
//
// The service scales to zero, so the first request after a while waits about a
// minute. Saved responses ship with the static site, so the examples answer at once
// and spend nothing. Run this by hand after the served corpus or prompt changes
// (it spends generation budget, so no build hook runs it) and commit the result:
//
//   node scripts/pregenerate-examples.mjs https://<service>
//
// It fails loudly on any error, leaving the previous file in place.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const EXAMPLES_FILE = resolve(HERE, "..", "src", "examples.json");
const OUT_FILE = resolve(HERE, "..", "src", "pregenerated-examples.json");

/** Enough for a cold start of the service. */
const TIMEOUT_MS = 180_000;

/** POST a JSON body to the service, failing on anything but a success. */
async function post(api, path, body) {
  const response = await fetch(`${api}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(
      `${path} returned ${response.status} for "${body.question ?? body.query}"`,
    );
  }
  return response.json();
}

const api = process.argv[2]?.replace(/\/+$/, "");
if (!api) {
  throw new Error("Usage: node scripts/pregenerate-examples.mjs https://<service>");
}

const examples = JSON.parse(readFileSync(EXAMPLES_FILE, "utf8"));
const saved = { generated: new Date().toISOString().slice(0, 10), ask: {}, search: {} };
for (const question of examples) {
  // The same k the frontend sends by default (src/api.ts).
  saved.ask[question] = await post(api, "/ask", { question, k: 5 });
  saved.search[question] = await post(api, "/search", { query: question, k: 10 });
  console.log(`Saved "${question}"`);
}
writeFileSync(OUT_FILE, `${JSON.stringify(saved, null, 2)}\n`);
console.log(`Wrote ${OUT_FILE}`);
