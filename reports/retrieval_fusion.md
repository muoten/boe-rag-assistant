# Fusion ablation — BM25 vs dense vs RRF (+ rerank)

- **Generated:** 2026-11-04 08:23 UTC
- **Corpus:** `boe-2015-present.parquet` (25920 chunks, 758 byte-identical duplicates folded)
- **Cached depth per leg:** 100
- **Served:** 50 per leg, k_rrf=10, no rerank (BOE_RRF_K, BOE_RERANK)
- **BM25 tokens:** norm numbers kept whole (`39/2015` is one token)
- **Dense embeddings:** `boe-embeddings-title.npz`
- **BM25 text:** document title + text; title twice on head chunks
- **Best fused variant:** rrf k60 d50 bm25x2 (chosen on silver-dev by MRR)
- **Dropped (gold not in corpus):** gold 0, silver 0
- **Silver split:** by document, test 0.5, seed 0

Recall@5 is what `/ask` grounds an answer in; recall@10 is what `/search`
shows. Values are means with 95% bootstrap intervals. *Reachable@d* is the
share of questions whose gold chunk is in either leg's top d — the ceiling
for any fusion or reranking over those candidates.

## gold (20 questions)

Reachable: @20 0.950, @50 1.000, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.850 [0.70, 1.00] | 0.850 [0.70, 1.00] | 0.809 [0.65, 0.95] | 0.813 [0.64, 0.95] |
| dense | 0.850 [0.70, 1.00] | 0.900 [0.75, 1.00] | 0.766 [0.59, 0.92] | 0.793 [0.63, 0.93] |
| rrf k10 d50 (served) | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.794 [0.64, 0.93] | 0.831 [0.70, 0.94] |
| rrf k60 d50 (upstream) | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.779 [0.62, 0.92] | 0.819 [0.68, 0.94] |
| rrf k60 d20 (ablation) | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.777 [0.62, 0.92] | 0.819 [0.68, 0.94] |
| rrf k60 d100 | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.779 [0.62, 0.92] | 0.819 [0.68, 0.94] |
| rrf k30 d50 | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.779 [0.62, 0.92] | 0.819 [0.68, 0.94] |
| rrf k100 d50 | 0.950 [0.85, 1.00] | 0.950 [0.85, 1.00] | 0.779 [0.62, 0.92] | 0.819 [0.68, 0.94] |
| rrf k60 d50 bm25x2 | 0.850 [0.70, 1.00] | 0.950 [0.85, 1.00] | 0.784 [0.63, 0.93] | 0.823 [0.68, 0.94] |
| rrf k60 d50 densex2 | 0.900 [0.75, 1.00] | 0.950 [0.85, 1.00] | 0.771 [0.61, 0.92] | 0.812 [0.67, 0.94] |
| rrf k10 d50 (served) + rerank | 0.950 [0.85, 1.00] | 1.000 [1.00, 1.00] | 0.870 [0.74, 0.97] | 0.901 [0.80, 0.98] |
| rrf k60 d50 (upstream) + rerank | 0.950 [0.85, 1.00] | 1.000 [1.00, 1.00] | 0.870 [0.74, 0.97] | 0.901 [0.80, 0.98] |
| rrf k60 d50 bm25x2 + rerank | 0.950 [0.85, 1.00] | 1.000 [1.00, 1.00] | 0.870 [0.74, 0.97] | 0.901 [0.80, 0.98] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | +0.000 | [-0.200, +0.200] | 1.000 |
| dense vs bm25 | mrr | -0.044 | [-0.242, +0.155] | 0.700 |
| rrf k10 d50 (served) vs bm25 | recall@5 | +0.100 | [+0.000, +0.250] | 0.508 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.016 | [-0.113, +0.074] | 0.720 |
| rrf k10 d50 (served) vs dense | recall@5 | +0.100 | [+0.000, +0.250] | 0.489 |
| rrf k10 d50 (served) vs dense | mrr | +0.028 | [-0.111, +0.165] | 0.713 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.015 | [-0.034, +0.000] | 0.254 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | +0.076 | [-0.079, +0.233] | 0.374 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | -0.017 | [-0.037, -0.002] | 0.129 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | -0.100 | [-0.250, +0.000] | 0.508 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | -0.010 | [-0.083, +0.064] | 0.692 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | +0.076 | [-0.079, +0.233] | 0.374 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.000 | [+0.000, +0.000] | 1.000 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.000 | [+0.000, +0.000] | 1.000 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| no legal reference | 20 | 0.850 / 0.809 | 0.850 / 0.766 | 0.950 / 0.794 | 0.950 / 0.870 |
| overlap low (<= 0.67) | 8 | 0.625 / 0.649 | 0.875 / 0.880 | 0.875 / 0.703 | 0.875 / 0.799 |
| overlap mid | 7 | 1.000 / 0.929 | 1.000 / 0.719 | 1.000 / 0.929 | 1.000 / 0.929 |
| overlap high (> 0.86) | 5 | 1.000 / 0.900 | 0.600 / 0.647 | 1.000 / 0.750 | 1.000 / 0.900 |

## norm-refs (14 questions)

Reachable: @20 1.000, @50 1.000, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.705 [0.54, 0.86] | 0.779 [0.66, 0.89] |
| dense | 0.357 [0.14, 0.64] | 0.500 [0.21, 0.79] | 0.135 [0.07, 0.22] | 0.216 [0.10, 0.34] |
| rrf k10 d50 (served) | 0.786 [0.57, 1.00] | 0.929 [0.79, 1.00] | 0.547 [0.36, 0.74] | 0.634 [0.47, 0.80] |
| rrf k60 d50 (upstream) | 0.571 [0.29, 0.79] | 0.714 [0.50, 0.93] | 0.500 [0.29, 0.71] | 0.535 [0.32, 0.75] |
| rrf k60 d20 (ablation) | 0.643 [0.36, 0.86] | 1.000 [1.00, 1.00] | 0.513 [0.32, 0.72] | 0.625 [0.47, 0.78] |
| rrf k60 d100 | 0.571 [0.29, 0.79] | 0.643 [0.36, 0.86] | 0.494 [0.29, 0.70] | 0.514 [0.29, 0.73] |
| rrf k30 d50 | 0.643 [0.36, 0.86] | 0.714 [0.50, 0.93] | 0.508 [0.30, 0.72] | 0.541 [0.33, 0.75] |
| rrf k100 d50 | 0.571 [0.29, 0.79] | 0.714 [0.50, 0.93] | 0.500 [0.29, 0.71] | 0.535 [0.32, 0.75] |
| rrf k60 d50 bm25x2 | 0.643 [0.36, 0.86] | 0.786 [0.57, 1.00] | 0.584 [0.36, 0.81] | 0.617 [0.40, 0.83] |
| rrf k60 d50 densex2 | 0.571 [0.29, 0.79] | 0.643 [0.36, 0.86] | 0.377 [0.19, 0.58] | 0.437 [0.24, 0.64] |
| rrf k10 d50 (served) + rerank | 0.429 [0.14, 0.71] | 0.714 [0.50, 0.93] | 0.175 [0.12, 0.23] | 0.285 [0.18, 0.38] |
| rrf k60 d50 (upstream) + rerank | 0.500 [0.21, 0.79] | 0.714 [0.50, 0.93] | 0.178 [0.13, 0.23] | 0.288 [0.18, 0.38] |
| rrf k60 d50 bm25x2 + rerank | 0.571 [0.29, 0.86] | 0.857 [0.64, 1.00] | 0.244 [0.17, 0.32] | 0.381 [0.28, 0.47] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | -0.643 | [-0.857, -0.357] | 0.005 |
| dense vs bm25 | mrr | -0.570 | [-0.751, -0.382] | 0.000 |
| rrf k10 d50 (served) vs bm25 | recall@5 | -0.214 | [-0.429, +0.000] | 0.253 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.158 | [-0.432, +0.116] | 0.292 |
| rrf k10 d50 (served) vs dense | recall@5 | +0.429 | [+0.143, +0.714] | 0.033 |
| rrf k10 d50 (served) vs dense | mrr | +0.412 | [+0.285, +0.545] | 0.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | -0.214 | [-0.429, +0.000] | 0.248 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.047 | [-0.089, -0.015] | 0.030 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | -0.286 | [-0.571, +0.000] | 0.220 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | -0.369 | [-0.549, -0.200] | 0.001 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | -0.143 | [-0.429, +0.143] | 0.629 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | -0.034 | [-0.097, +0.005] | 0.345 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | -0.143 | [-0.357, +0.000] | 0.501 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | +0.037 | [-0.055, +0.153] | 0.504 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | -0.357 | [-0.643, +0.000] | 0.128 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | -0.372 | [-0.552, -0.203] | 0.001 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.143 | [+0.000, +0.357] | 0.503 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.069 | [+0.017, +0.134] | 0.015 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 14 | 1.000 / 0.705 | 0.357 / 0.135 | 0.786 / 0.547 | 0.429 / 0.175 |
| no legal reference | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| overlap low (<= 0.50) | 13 | 1.000 / 0.744 | 0.385 / 0.141 | 0.769 / 0.551 | 0.385 / 0.163 |
| overlap mid | 0 | nan / nan | nan / nan | nan / nan | nan / nan |
| overlap high (> 0.50) | 1 | 1.000 / 0.200 | 0.000 / 0.048 | 1.000 / 0.500 | 1.000 / 0.333 |

## silver-dev (866 questions)

Reachable: @20 0.990, @50 0.999, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.935 [0.92, 0.95] | 0.967 [0.95, 0.98] | 0.845 [0.83, 0.86] | 0.874 [0.86, 0.89] |
| dense | 0.893 [0.87, 0.91] | 0.941 [0.92, 0.96] | 0.780 [0.76, 0.80] | 0.817 [0.80, 0.84] |
| rrf k10 d50 (served) | 0.952 [0.94, 0.97] | 0.979 [0.97, 0.99] | 0.840 [0.82, 0.86] | 0.874 [0.86, 0.89] |
| rrf k60 d50 (upstream) | 0.943 [0.93, 0.96] | 0.978 [0.97, 0.99] | 0.836 [0.82, 0.85] | 0.870 [0.85, 0.89] |
| rrf k60 d20 (ablation) | 0.940 [0.92, 0.95] | 0.975 [0.96, 0.98] | 0.834 [0.81, 0.85] | 0.868 [0.85, 0.88] |
| rrf k60 d100 | 0.943 [0.93, 0.96] | 0.979 [0.97, 0.99] | 0.835 [0.82, 0.85] | 0.870 [0.85, 0.89] |
| rrf k30 d50 | 0.943 [0.93, 0.96] | 0.977 [0.97, 0.99] | 0.837 [0.82, 0.86] | 0.871 [0.85, 0.89] |
| rrf k100 d50 | 0.942 [0.93, 0.96] | 0.976 [0.97, 0.98] | 0.835 [0.82, 0.85] | 0.869 [0.85, 0.89] |
| rrf k60 d50 bm25x2 | 0.949 [0.93, 0.96] | 0.978 [0.97, 0.99] | 0.856 [0.84, 0.87] | 0.885 [0.87, 0.90] |
| rrf k60 d50 densex2 | 0.933 [0.92, 0.95] | 0.964 [0.95, 0.98] | 0.826 [0.80, 0.85] | 0.859 [0.84, 0.88] |

## silver-test (883 questions)

Reachable: @20 0.992, @50 0.999, @100 1.000

| System | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| bm25 | 0.945 [0.93, 0.96] | 0.971 [0.96, 0.98] | 0.865 [0.85, 0.88] | 0.890 [0.87, 0.91] |
| dense | 0.904 [0.88, 0.92] | 0.946 [0.93, 0.96] | 0.798 [0.78, 0.82] | 0.832 [0.81, 0.85] |
| rrf k10 d50 (served) | 0.951 [0.94, 0.96] | 0.978 [0.97, 0.99] | 0.863 [0.84, 0.88] | 0.891 [0.88, 0.91] |
| rrf k60 d50 (upstream) | 0.950 [0.94, 0.96] | 0.975 [0.96, 0.99] | 0.860 [0.84, 0.88] | 0.888 [0.87, 0.90] |
| rrf k60 d20 (ablation) | 0.948 [0.93, 0.96] | 0.977 [0.97, 0.99] | 0.860 [0.84, 0.88] | 0.888 [0.87, 0.90] |
| rrf k60 d100 | 0.950 [0.94, 0.96] | 0.977 [0.97, 0.99] | 0.860 [0.84, 0.88] | 0.889 [0.87, 0.90] |
| rrf k30 d50 | 0.951 [0.94, 0.96] | 0.976 [0.97, 0.99] | 0.861 [0.84, 0.88] | 0.889 [0.87, 0.90] |
| rrf k100 d50 | 0.949 [0.93, 0.96] | 0.974 [0.96, 0.98] | 0.860 [0.84, 0.88] | 0.888 [0.87, 0.90] |
| rrf k60 d50 bm25x2 | 0.956 [0.94, 0.97] | 0.977 [0.97, 0.99] | 0.871 [0.85, 0.89] | 0.897 [0.88, 0.91] |
| rrf k60 d50 densex2 | 0.945 [0.93, 0.96] | 0.966 [0.95, 0.98] | 0.848 [0.83, 0.87] | 0.877 [0.86, 0.89] |
| rrf k10 d50 (served) + rerank | 0.929 [0.91, 0.95] | 0.975 [0.96, 0.99] | 0.791 [0.77, 0.81] | 0.836 [0.82, 0.85] |
| rrf k60 d50 (upstream) + rerank | 0.932 [0.92, 0.95] | 0.976 [0.97, 0.99] | 0.793 [0.77, 0.81] | 0.837 [0.82, 0.85] |
| rrf k60 d50 bm25x2 + rerank | 0.929 [0.91, 0.95] | 0.974 [0.96, 0.98] | 0.793 [0.77, 0.81] | 0.837 [0.82, 0.85] |

| Candidate vs baseline | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| dense vs bm25 | recall@5 | -0.041 | [-0.060, -0.023] | 0.000 |
| dense vs bm25 | mrr | -0.067 | [-0.088, -0.046] | 0.000 |
| rrf k10 d50 (served) vs bm25 | recall@5 | +0.007 | [-0.006, +0.019] | 0.384 |
| rrf k10 d50 (served) vs bm25 | mrr | -0.002 | [-0.017, +0.013] | 0.771 |
| rrf k10 d50 (served) vs dense | recall@5 | +0.048 | [+0.033, +0.063] | 0.000 |
| rrf k10 d50 (served) vs dense | mrr | +0.065 | [+0.052, +0.078] | 0.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | recall@5 | -0.001 | [-0.008, +0.006] | 1.000 |
| rrf k60 d50 (upstream) vs rrf k10 d50 (served) | mrr | -0.002 | [-0.005, +0.000] | 0.062 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | recall@5 | -0.019 | [-0.037, -0.002] | 0.045 |
| rrf k60 d50 (upstream) + rerank vs rrf k10 d50 (served) | mrr | -0.069 | [-0.092, -0.047] | 0.000 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | recall@5 | -0.003 | [-0.010, +0.003] | 0.507 |
| rrf k60 d20 (ablation) vs rrf k10 d50 (served) | mrr | -0.003 | [-0.006, -0.000] | 0.043 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | recall@5 | +0.005 | [-0.001, +0.011] | 0.287 |
| rrf k60 d50 bm25x2 vs rrf k10 d50 (served) | mrr | +0.009 | [-0.002, +0.019] | 0.104 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | recall@5 | -0.023 | [-0.041, -0.005] | 0.019 |
| rrf k10 d50 (served) + rerank vs rrf k10 d50 (served) | mrr | -0.071 | [-0.094, -0.049] | 0.000 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | recall@5 | +0.000 | [-0.007, +0.007] | 1.000 |
| rrf k60 d50 bm25x2 + rerank vs rrf k10 d50 (served) + rerank | mrr | +0.001 | [-0.002, +0.005] | 0.400 |

| Slice | n | bm25 R@5 / MRR | dense R@5 / MRR | rrf k10 d50 (served) R@5 / MRR | rrf k10 d50 (served) + rerank R@5 / MRR |
|---|---|---|---|---|---|
| legal reference | 740 | 0.947 / 0.876 | 0.909 / 0.808 | 0.958 / 0.877 | 0.924 / 0.788 |
| no legal reference | 143 | 0.930 / 0.809 | 0.874 / 0.742 | 0.916 / 0.790 | 0.951 / 0.809 |
| overlap low (<= 0.60) | 315 | 0.914 / 0.813 | 0.873 / 0.777 | 0.924 / 0.840 | 0.886 / 0.728 |
| overlap mid | 284 | 0.961 / 0.892 | 0.923 / 0.807 | 0.961 / 0.871 | 0.944 / 0.793 |
| overlap high (> 0.72) | 284 | 0.961 / 0.894 | 0.919 / 0.811 | 0.972 / 0.879 | 0.961 / 0.860 |
