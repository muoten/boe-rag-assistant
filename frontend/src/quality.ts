// Typed access to the curated quality dataset.
//
// quality-data.json is generated from reports/*.json by scripts/build-quality-data.mjs
// (run via the prebuild/predev npm hooks). It is imported at build time, so the
// quality page needs no network request and cannot fail to load.

import data from "./quality-data.json";
import type { QualityData } from "./types";

export const quality = data as QualityData;

/** Format an ISO date in long Spanish form (e.g. "1 de enero de 2015"). */
export const longDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("es-ES", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });

/** The knowledge-base sentence shown on both views. */
export const corpusScope = `una parte del BOE: ${quality.corpus.documents.toLocaleString("es-ES")} documentos publicados entre el ${longDate(quality.corpus.first_published)} y el ${longDate(quality.corpus.last_published)}`;
