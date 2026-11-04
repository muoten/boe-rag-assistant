import { quality } from "../quality";
import type { MetricCI, RetrievalStats } from "../types";
import { AblationTable } from "./AblationTable";
import { MetricCard } from "./MetricCard";

/** Format a 0–1 ratio as a whole-ish percentage (e.g. 0.913 → "91.3 %"). */
const pct = (value: number) => `${(value * 100).toFixed(1)} %`;

/** Render a confidence interval as a compact "IC 95% lo–hi %" detail string. */
function ciDetail(ci: MetricCI | null): string | undefined {
  if (!ci) {
    return undefined;
  }
  return `IC 95 % ${pct(ci.low)} – ${pct(ci.high)}`;
}

/** A retrieval eval set's three headline metrics as a row of cards. */
function RetrievalCards({ stats }: { stats: RetrievalStats }) {
  const recallDetail = ciDetail(stats.recall_ci);
  const mrrDetail = ciDetail(stats.mrr_ci);
  return (
    <div className="metric-grid">
      <MetricCard
        label="Recall@5"
        value={pct(stats.recall_at_5)}
        {...(recallDetail ? { detail: recallDetail } : {})}
      />
      <MetricCard
        label="MRR"
        value={stats.mrr.toFixed(3)}
        {...(mrrDetail ? { detail: mrrDetail } : {})}
      />
      <MetricCard label="nDCG@10" value={stats.ndcg_at_10.toFixed(3)} />
    </div>
  );
}

/** The about page: credits, then headline eval metrics from the committed reports. */
export function QualityPage() {
  const { retrieval, ablation, generation, security } = quality;
  const measuredDate = new Date(quality.retrieval_measured).toLocaleDateString(
    "es-ES",
    {
      year: "numeric",
      month: "long",
      day: "numeric",
    },
  );

  return (
    <div className="quality">
      <section className="quality-section">
        <h2>Créditos</h2>
        <p className="quality-note">
          Este proyecto parte de{" "}
          <a href="https://github.com/gonzalonao/boe-rag-assistant">
            boe-rag-assistant
          </a>
          , de Gonzalo López Crespo, publicado con licencia MIT. Esta versión (
          <a href="https://github.com/muoten/boe-rag-assistant">
            muoten/boe-rag-assistant
          </a>
          ) lo despliega en Azure y modifica la recuperación.
        </p>
      </section>

      <section className="quality-section">
        <h2>Calidad</h2>
        <p className="quality-intro">
          Cada cifra procede de un <em>eval</em> reproducible versionado en el
          repositorio (<code>reports/</code>); no son estimaciones. La recuperación se
          mide con equivalencia textual; las respuestas, con un juez LLM.
        </p>
      </section>

      <section className="quality-section">
        <h2>Recuperación</h2>
        <p className="quality-note">
          Lo que se sirve en producción, medido el {measuredDate}. Recall@5 cuenta los
          pasajes en los que se fundamenta cada respuesta.
        </p>
        <h3 className="quality-subhead">
          Conjunto oro · {retrieval.gold.num_queries} preguntas curadas a mano
        </h3>
        <RetrievalCards stats={retrieval.gold} />
        <h3 className="quality-subhead">
          Conjunto plata (test) · {retrieval.silver.num_queries.toLocaleString("es-ES")}{" "}
          preguntas generadas automáticamente
        </h3>
        <RetrievalCards stats={retrieval.silver} />
        <h3 className="quality-subhead">
          Referencias normativas · {retrieval.norm_refs.num_queries} preguntas «¿Qué
          regula la norma X?»
        </h3>
        <RetrievalCards stats={retrieval.norm_refs} />
      </section>

      <section className="quality-section">
        <h2>Pipeline de recuperación</h2>
        <p className="quality-note">
          En producción se fusionan BM25 y búsqueda densa con RRF,{" "}
          <strong>sin reordenar</strong>. El <strong>cross-encoder</strong> empeoró la
          recuperación en el conjunto plata y en las referencias normativas, y añade
          latencia; en el conjunto oro mejora el MRR, sin diferencia estadísticamente
          significativa. En las referencias normativas, BM25 por sí solo supera al
          híbrido servido.
        </p>
        <h3 className="quality-subhead">Conjunto oro</h3>
        <AblationTable rows={ablation.gold} />
        <h3 className="quality-subhead">Conjunto plata (test)</h3>
        <AblationTable rows={ablation.silver} />
        <h3 className="quality-subhead">Referencias normativas</h3>
        <AblationTable rows={ablation.norm_refs} />
      </section>

      <section className="quality-section">
        <h2>Calidad de las respuestas</h2>
        <p className="quality-note">
          Juez LLM sobre {generation.num_queries} preguntas de extremo a extremo. Medido
          en julio de 2026, antes de los cambios actuales en la recuperación y sin
          repetir desde entonces.
        </p>
        <div className="metric-grid">
          <MetricCard label="Fidelidad" value={pct(generation.mean_faithfulness)} />
          <MetricCard label="Corrección" value={pct(generation.mean_correctness)} />
          <MetricCard
            label="Tasa de rechazo"
            value={pct(generation.refusal_rate)}
            detail="rechaza cuando no hay base"
          />
        </div>
      </section>

      <section className="quality-section">
        <h2>Robustez adversarial</h2>
        <p className="quality-note">
          {security.num_passed} de {security.num_cases} casos superados (
          {pct(security.pass_rate)}). Medido en julio de 2026, sin repetir desde
          entonces.
        </p>
        <ul className="security-list">
          {security.by_category.map((cat) => (
            <li key={cat.name} className="security-row">
              <span className="security-name">{cat.name.replace(/_/g, " ")}</span>
              <span className="security-bar" aria-hidden="true">
                <span
                  className="security-bar-fill"
                  style={{ width: `${cat.pass_rate * 100}%` }}
                />
              </span>
              <span className="security-pct">{pct(cat.pass_rate)}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
