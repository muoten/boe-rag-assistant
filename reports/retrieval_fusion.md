# Fusion ablation — BM25 vs dense vs RRF (+ rerank)

- **Generated:** 2026-11-03 10:58 UTC
- **Corpus:** `boe-2015-present.parquet` (25920 chunks, 758 byte-identical duplicates folded)
- **Cached depth per leg:** 100
- **Served:** 50 per leg, k_rrf=10, no rerank (BOE_RRF_K, BOE_RERANK)
- **BM25 tokens:** norm numbers kept whole (`39/2015` is one token)
- **BM25 text:** document title + text; title twice on head chunks
- **Best fused variant:** rrf k60 d50 bm25x2 (chosen on silver-dev by MRR)
- **Dropped (gold not in corpus):** gold 0, silver 0
- **Silver split:** by document, test 0.5, seed 0

Recall@5 is what `/ask` grounds an answer in; recall@10 is what `/search`
shows. Values are means with 95% bootstrap intervals. *Reachable@d* is the
share of questions whose gold chunk is in either leg's top d — the ceiling
for any fusion or reranking over those candidates.

## gold (20 questions)

Reachable: @20 1.000, @50 1.000, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.850 [0.70, 1.00] | 0.850 [0.70, 1.00] | 0.809 [0.65, 0.95] | 0.813 [0.64, 0.95] |
| dense | 0.900 [0.75, 1.00] | 0.900 [0.75, 1.00] | 0.691 [0.52, 0.84] | 0.740 [0.58, 0.88] |
| rrf k10 d50 (served) | 0.850 [0.70, 1.00] | 0.950 [0.85, 1.00] | 0.724 [0.56, 0.87] | 0.777 [0.64, 0.90] |
| rrf k60 d50 (upstream) | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.718 [0.55, 0.87] | 0.759 [0.60, 0.89] |
| rrf k60 d20 (ablation) | 0.900 [0.75, 1.00] | 0.950 [0.85, 1.00] | 0.728 [0.57, 0.87] | 0.781 [0.64, 0.90] |
| rrf k60 d100 | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.717 [0.55, 0.87] | 0.759 [0.60, 0.89] |
| rrf k30 d50 | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.719 [0.56, 0.87] | 0.759 [0.60, 0.89] |
| rrf k100 d50 | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.718 [0.55, 0.87] | 0.759 [0.60, 0.89] |
| rrf k60 d50 bm25x2 | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.770 [0.60, 0.92] | 0.796 [0.63, 0.93] |
| rrf k60 d50 densex2 | 0.900 [0.75, 1.00] | 0.900 [0.75, 1.00] | 0.710 [0.54, 0.86] | 0.754 [0.60, 0.89] |
| rrf k10 d50 (served) + rerank | 0.900 [0.75, 1.00] | 1.000 [1.00, 1.00] | 0.864 [0.73, 0.97] | 0.896 [0.79, 0.98] |
| rrf k60 d50 (upstream) + rerank | 0.900 [0.75, 1.00] | 1.000 [1.00, 1.00] | 0.864 [0.73, 0.97] | 0.896 [0.79, 0.98] |
| rrf k60 d50 bm25x2 + rerank | 0.900 [0.75, 1.00] | 1.000 [1.00, 1.00] | 0.865 [0.73, 0.97] | 0.898 [0.79, 0.98] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | +0.050 | [-0.100, +0.200] | 1.000 |
| dense vs bm25 | mrr | -0.119 | [-0.314, +0.075] | 0.264 |
| rrf k10 d50 (served) vs bm25 | recall@5 | +0.000 | [-0.150, +0.150] | 1.000 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.086 | [-0.217, +0.036] | 0.245 |
| rrf k10 d50 (served) vs dense | recall@5 | -0.050 | [-0.150, +0.000] | 1.000 |
| rrf k10 d50 (served) vs dense | mrr | +0.033 | [-0.091, +0.150] | 0.645 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.005 | [-0.016, +0.000] | 1.000 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | +0.050 | [+0.000, +0.150] | 1.000 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | +0.141 | [-0.016, +0.301] | 0.112 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | +0.050 | [+0.000, +0.150] | 1.000 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | +0.004 | [+0.000, +0.013] | 1.000 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | +0.047 | [-0.050, +0.146] | 0.465 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | +0.050 | [+0.000, +0.150] | 1.000 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | +0.141 | [-0.016, +0.301] | 0.112 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.001 | [+0.000, +0.004] | 1.000 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| no legal reference | 20 | 0.850 / 0.809 | 0.900 / 0.691 | 0.850 / 0.724 | 0.900 / 0.864 |
| overlap low (<= 0.67) | 8 | 0.625 / 0.649 | 0.750 / 0.635 | 0.625 / 0.528 | 0.750 / 0.786 |
| overlap mid | 7 | 1.000 / 0.929 | 1.000 / 0.762 | 1.000 / 0.929 | 1.000 / 0.929 |
| overlap high (> 0.86) | 5 | 1.000 / 0.900 | 1.000 / 0.680 | 1.000 / 0.750 | 1.000 / 0.900 |

## norm-refs (14 questions)

Reachable: @20 1.000, @50 1.000, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.705 [0.54, 0.86] | 0.779 [0.66, 0.89] |
| dense | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.001 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| rrf k10 d50 (served) | 0.786 [0.57, 1.00] | 0.929 [0.79, 1.00] | 0.294 [0.23, 0.36] | 0.443 [0.35, 0.52] |
| rrf k60 d50 (upstream) | 0.643 [0.36, 0.86] | 0.929 [0.79, 1.00] | 0.210 [0.17, 0.27] | 0.373 [0.30, 0.44] |
| rrf k60 d20 (ablation) | 0.786 [0.57, 1.00] | 0.929 [0.79, 1.00] | 0.314 [0.24, 0.39] | 0.460 [0.37, 0.54] |
| rrf k60 d100 | 0.357 [0.14, 0.64] | 1.000 [1.00, 1.00] | 0.165 [0.14, 0.19] | 0.352 [0.33, 0.38] |
| rrf k30 d50 | 0.643 [0.36, 0.86] | 0.929 [0.79, 1.00] | 0.211 [0.17, 0.27] | 0.373 [0.30, 0.44] |
| rrf k100 d50 | 0.643 [0.36, 0.86] | 0.929 [0.79, 1.00] | 0.210 [0.17, 0.27] | 0.373 [0.30, 0.44] |
| rrf k60 d50 bm25x2 | 0.786 [0.57, 1.00] | 1.000 [1.00, 1.00] | 0.326 [0.24, 0.45] | 0.488 [0.42, 0.58] |
| rrf k60 d50 densex2 | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.020 [0.02, 0.02] | 0.000 [0.00, 0.00] |
| rrf k10 d50 (served) + rerank | 0.143 [0.00, 0.36] | 0.500 [0.21, 0.79] | 0.122 [0.09, 0.17] | 0.179 [0.08, 0.28] |
| rrf k60 d50 (upstream) + rerank | 0.143 [0.00, 0.36] | 0.500 [0.21, 0.79] | 0.122 [0.09, 0.17] | 0.179 [0.08, 0.28] |
| rrf k60 d50 bm25x2 + rerank | 0.500 [0.21, 0.79] | 0.929 [0.79, 1.00] | 0.238 [0.17, 0.31] | 0.396 [0.31, 0.47] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | -1.000 | [-1.000, -1.000] | 0.000 |
| dense vs bm25 | mrr | -0.704 | [-0.857, -0.540] | 0.000 |
| rrf k10 d50 (served) vs bm25 | recall@5 | -0.214 | [-0.429, +0.000] | 0.249 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.411 | [-0.513, -0.308] | 0.000 |
| rrf k10 d50 (served) vs dense | recall@5 | +0.786 | [+0.571, +1.000] | 0.001 |
| rrf k10 d50 (served) vs dense | mrr | +0.293 | [+0.224, +0.364] | 0.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | -0.143 | [-0.357, +0.000] | 0.505 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.084 | [-0.133, -0.041] | 0.001 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | -0.643 | [-0.929, -0.286] | 0.011 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | -0.172 | [-0.257, -0.081] | 0.004 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | +0.020 | [-0.020, +0.061] | 0.346 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | +0.000 | [-0.214, +0.214] | 1.000 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | +0.032 | [-0.035, +0.121] | 0.562 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | -0.643 | [-0.929, -0.286] | 0.011 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | -0.172 | [-0.257, -0.081] | 0.004 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.357 | [+0.143, +0.643] | 0.062 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.116 | [+0.049, +0.195] | 0.002 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 14 | 1.000 / 0.705 | 0.000 / 0.001 | 0.786 / 0.294 | 0.143 / 0.122 |
| no legal reference | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| overlap low (<= 0.50) | 13 | 1.000 / 0.744 | 0.000 / 0.000 | 0.846 / 0.310 | 0.077 / 0.106 |
| overlap mid | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| overlap high (> 0.50) | 1 | 1.000 / 0.200 | 0.000 / 0.013 | 0.000 / 0.091 | 1.000 / 0.333 |

## silver-dev (866 questions)

Reachable: @20 0.990, @50 0.999, @100 0.999

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.935 [0.92, 0.95] | 0.967 [0.95, 0.98] | 0.845 [0.83, 0.86] | 0.874 [0.86, 0.89] |
| dense | 0.871 [0.85, 0.89] | 0.924 [0.91, 0.94] | 0.759 [0.73, 0.78] | 0.796 [0.77, 0.82] |
| rrf k10 d50 (served) | 0.949 [0.93, 0.96] | 0.978 [0.97, 0.99] | 0.842 [0.82, 0.86] | 0.875 [0.86, 0.89] |
| rrf k60 d50 (upstream) | 0.945 [0.93, 0.96] | 0.973 [0.96, 0.98] | 0.838 [0.82, 0.86] | 0.870 [0.85, 0.89] |
| rrf k60 d20 (ablation) | 0.946 [0.93, 0.96] | 0.971 [0.96, 0.98] | 0.837 [0.82, 0.86] | 0.869 [0.85, 0.89] |
| rrf k60 d100 | 0.948 [0.93, 0.96] | 0.971 [0.96, 0.98] | 0.838 [0.82, 0.86] | 0.870 [0.85, 0.89] |
| rrf k30 d50 | 0.946 [0.93, 0.96] | 0.973 [0.96, 0.98] | 0.838 [0.82, 0.86] | 0.871 [0.85, 0.89] |
| rrf k100 d50 | 0.943 [0.93, 0.96] | 0.973 [0.96, 0.98] | 0.838 [0.82, 0.86] | 0.870 [0.85, 0.89] |
| rrf k60 d50 bm25x2 | 0.946 [0.93, 0.96] | 0.977 [0.97, 0.99] | 0.849 [0.83, 0.87] | 0.880 [0.86, 0.90] |
| rrf k60 d50 densex2 | 0.934 [0.92, 0.95] | 0.962 [0.95, 0.97] | 0.824 [0.80, 0.84] | 0.857 [0.84, 0.87] |

## silver-test (883 questions)

Reachable: @20 0.991, @50 1.000, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.945 [0.93, 0.96] | 0.971 [0.96, 0.98] | 0.865 [0.85, 0.88] | 0.890 [0.87, 0.91] |
| dense | 0.890 [0.87, 0.91] | 0.928 [0.91, 0.94] | 0.778 [0.76, 0.80] | 0.813 [0.79, 0.83] |
| rrf k10 d50 (served) | 0.954 [0.94, 0.97] | 0.976 [0.97, 0.99] | 0.854 [0.84, 0.87] | 0.884 [0.87, 0.90] |
| rrf k60 d50 (upstream) | 0.952 [0.94, 0.97] | 0.973 [0.96, 0.98] | 0.853 [0.83, 0.87] | 0.882 [0.87, 0.90] |
| rrf k60 d20 (ablation) | 0.951 [0.94, 0.96] | 0.973 [0.96, 0.98] | 0.853 [0.83, 0.87] | 0.882 [0.87, 0.90] |
| rrf k60 d100 | 0.952 [0.94, 0.97] | 0.975 [0.96, 0.99] | 0.853 [0.83, 0.87] | 0.882 [0.87, 0.90] |
| rrf k30 d50 | 0.952 [0.94, 0.97] | 0.974 [0.96, 0.98] | 0.854 [0.84, 0.87] | 0.883 [0.87, 0.90] |
| rrf k100 d50 | 0.952 [0.94, 0.97] | 0.973 [0.96, 0.98] | 0.853 [0.83, 0.87] | 0.882 [0.87, 0.90] |
| rrf k60 d50 bm25x2 | 0.956 [0.94, 0.97] | 0.974 [0.96, 0.98] | 0.869 [0.85, 0.89] | 0.894 [0.88, 0.91] |
| rrf k60 d50 densex2 | 0.939 [0.92, 0.95] | 0.964 [0.95, 0.98] | 0.838 [0.82, 0.86] | 0.869 [0.85, 0.89] |
| rrf k10 d50 (served) + rerank | 0.920 [0.90, 0.94] | 0.968 [0.96, 0.98] | 0.783 [0.76, 0.80] | 0.828 [0.81, 0.85] |
| rrf k60 d50 (upstream) + rerank | 0.920 [0.90, 0.94] | 0.967 [0.95, 0.98] | 0.784 [0.76, 0.81] | 0.828 [0.81, 0.85] |
| rrf k60 d50 bm25x2 + rerank | 0.928 [0.91, 0.94] | 0.971 [0.96, 0.98] | 0.788 [0.77, 0.81] | 0.832 [0.81, 0.85] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | -0.054 | [-0.075, -0.034] | 0.000 |
| dense vs bm25 | mrr | -0.087 | [-0.109, -0.065] | 0.000 |
| rrf k10 d50 (served) vs bm25 | recall@5 | +0.009 | [-0.003, +0.023] | 0.230 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.010 | [-0.026, +0.005] | 0.188 |
| rrf k10 d50 (served) vs dense | recall@5 | +0.063 | [+0.048, +0.080] | 0.000 |
| rrf k10 d50 (served) vs dense | mrr | +0.076 | [+0.062, +0.091] | 0.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | -0.001 | [-0.008, +0.006] | 1.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.001 | [-0.004, +0.001] | 0.368 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | -0.034 | [-0.052, -0.016] | 0.001 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | -0.071 | [-0.092, -0.049] | 0.000 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | -0.002 | [-0.009, +0.005] | 0.765 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | -0.002 | [-0.005, +0.001] | 0.233 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | +0.002 | [-0.006, +0.010] | 0.770 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | +0.015 | [+0.004, +0.025] | 0.008 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | -0.034 | [-0.052, -0.016] | 0.001 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | -0.071 | [-0.093, -0.050] | 0.000 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.008 | [+0.001, +0.016] | 0.065 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.005 | [+0.001, +0.009] | 0.008 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 740 | 0.947 / 0.876 | 0.886 / 0.776 | 0.957 / 0.862 | 0.915 / 0.779 |
| no legal reference | 143 | 0.930 / 0.809 | 0.909 / 0.791 | 0.937 / 0.816 | 0.944 / 0.807 |
| overlap low (<= 0.60) | 315 | 0.914 / 0.813 | 0.816 / 0.700 | 0.914 / 0.805 | 0.873 / 0.718 |
| overlap mid | 284 | 0.961 / 0.892 | 0.912 / 0.796 | 0.972 / 0.873 | 0.937 / 0.782 |
| overlap high (> 0.72) | 284 | 0.961 / 0.894 | 0.951 / 0.847 | 0.979 / 0.891 | 0.954 / 0.857 |
