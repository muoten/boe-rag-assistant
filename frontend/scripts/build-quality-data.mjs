// Assemble the quality page's data from the committed eval reports.
//
// The SPA is a static GitHub Pages site with no backend, so it cannot read the
// repo's reports/ at runtime. This script (run via the `prebuild`/`predev` npm
// hooks) curates the headline numbers from reports/*.json into a single typed
// JSON module that the page imports at build time. reports/ stays the single
// source of truth; the deployed page always reflects the latest committed eval.
//
// It fails loudly if a required report or field is missing, so a blank metrics
// page can never ship silently.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPORTS_DIR = resolve(HERE, "..", "..", "reports");
const OUT_FILE = resolve(HERE, "..", "src", "quality-data.json");

/** Read and parse a report JSON, failing with a clear message if absent. */
function readReport(name) {
  const path = resolve(REPORTS_DIR, name);
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`Cannot read required report ${name} at ${path}: ${err.message}`);
  }
}

/** Assert a field exists on a report object (numbers may legitimately be 0). */
function field(report, key, where) {
  const value = report[key];
  if (value === undefined || value === null) {
    throw new Error(`Report ${where} is missing required field "${key}".`);
  }
  return value;
}

// The fusion report's system names, mapped to the page's labels. Its served entry is
// the configuration deployed on Azure (see the report's "Served" metadata).
const SERVED = "rrf k10 d50 (served)";
const SYSTEMS = [
  ["bm25", "BM25"],
  ["dense", "Denso (e5)"],
  [SERVED, "Híbrido RRF (en producción)"],
  [`${SERVED} + rerank`, "Híbrido + cross-encoder (no servido)"],
];

/** A system's summary in one evaluation set of the fusion report. */
function systemSummary(set, key, where) {
  const summary = field(set, "summary", where)[key];
  if (!summary) {
    throw new Error(`Report ${where} is missing the "${key}" system.`);
  }
  return summary;
}

/** Project one evaluation set into the page's headline stats for the served system. */
function retrievalStats(report, name) {
  const where = `retrieval_fusion.${name}`;
  const set = field(report.sets, name, "retrieval_fusion");
  const summary = systemSummary(set, SERVED, where);
  const point = (metric) => field(field(summary, metric, where), "point", where);
  const ci = (metric) => ({ low: summary[metric].low, high: summary[metric].high });
  return {
    num_queries: field(set, "n", where),
    recall_at_5: point("recall@5"),
    mrr: point("mrr"),
    ndcg_at_10: point("ndcg@10"),
    recall_ci: ci("recall@5"),
    mrr_ci: ci("mrr"),
  };
}

/** The systems of one evaluation set, side by side. */
function ablation(report, name) {
  const where = `retrieval_fusion.${name}`;
  const set = field(report.sets, name, "retrieval_fusion");
  return SYSTEMS.map(([key, label]) => {
    const summary = systemSummary(set, key, where);
    return {
      name: label,
      served: key === SERVED,
      recall_at_5: summary["recall@5"].point,
      mrr: summary.mrr.point,
      ndcg_at_10: summary["ndcg@10"].point,
    };
  });
}

function security(report) {
  const byCategory = field(report, "pass_rate_by_category", "security_eval");
  return {
    num_cases: field(report, "num_cases", "security_eval"),
    num_passed: field(report, "num_passed", "security_eval"),
    pass_rate: field(report, "pass_rate", "security_eval"),
    by_category: Object.entries(byCategory)
      .map(([name, pass_rate]) => ({ name, pass_rate }))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
}

const fusion = readReport("retrieval_fusion.json");
const corpus = readReport("corpus_summary.json");
const e2e = readReport("e2e_baseline.json");
const sec = readReport("security_eval.json");

const data = {
  corpus: {
    documents: field(corpus, "documents", "corpus_summary"),
    passages: field(corpus, "passages", "corpus_summary"),
    first_published: field(corpus, "first_published", "corpus_summary"),
    last_published: field(corpus, "last_published", "corpus_summary"),
    section_one_sample: field(corpus, "section_one_sample", "corpus_summary"),
  },
  retrieval_measured: field(fusion.meta, "Generated", "retrieval_fusion.meta")
    .replace(" ", "T")
    .replace(" UTC", ":00Z"),
  retrieval: {
    gold: retrievalStats(fusion, "gold"),
    silver: retrievalStats(fusion, "silver-test"),
    norm_refs: retrievalStats(fusion, "norm-refs"),
  },
  ablation: {
    gold: ablation(fusion, "gold"),
    silver: ablation(fusion, "silver-test"),
    norm_refs: ablation(fusion, "norm-refs"),
  },
  generation: {
    num_queries: field(e2e, "num_queries", "e2e_baseline"),
    mean_faithfulness: field(e2e, "mean_faithfulness", "e2e_baseline"),
    mean_correctness: field(e2e, "mean_correctness", "e2e_baseline"),
    refusal_rate: field(e2e, "refusal_rate", "e2e_baseline"),
  },
  security: security(sec),
};

writeFileSync(OUT_FILE, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(
  `Wrote ${OUT_FILE} (${data.security.num_cases} security cases, ` +
    `${data.retrieval.silver.num_queries} silver queries).`,
);
