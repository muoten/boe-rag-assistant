import type { AblationRow } from "../types";

interface AblationTableProps {
  rows: AblationRow[];
}

const format = (value: number) => value.toFixed(3);

/** The retrieval ablation as a table: each system against its metrics.
 *
 * The system actually served is highlighted, so a reader can compare it with its
 * two legs and with the reranker it runs without.
 */
export function AblationTable({ rows }: AblationTableProps) {
  return (
    <div className="table-wrap">
      <table className="metric-table">
        <thead>
          <tr>
            <th scope="col">Sistema</th>
            <th scope="col">Recall@5</th>
            <th scope="col">MRR</th>
            <th scope="col">nDCG@10</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name} className={row.served ? "served" : undefined}>
              <th scope="row">{row.name}</th>
              <td>{format(row.recall_at_5)}</td>
              <td>{format(row.mrr)}</td>
              <td>{format(row.ndcg_at_10)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
