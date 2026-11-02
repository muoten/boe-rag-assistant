# Prompt A/B: partial answers vs. strict refusal (gpt-5-mini)

- **Generated:** 2026-11-02, `tools/boe_prompt_ab.py` in [muoten/azure-rag-observability](https://github.com/muoten/azure-rag-observability)
- **Generator and judge:** gpt-5-mini (Microsoft Foundry, `reasoning_effort: minimal`), the same model judging its own answers
- **Retrieval:** the live Azure `/search` (hybrid + cross-encoder rerank, k=5), fetched once per question and shared by both prompts
- **Answer path:** `generate_answer` + canary screen + cite-or-refuse validation, as in `RagEngine.answer`
- **Runs:** one per gold question; three per probe; one per adversarial case

**A** is the original refusal clause ("Si la respuesta no aparece en los fragmentos, responde exactamente…").
**B** answers from what the passages say when they concern what was asked, naming what is missing, and refuses when
they concern another law, tax, body or scope even if the wording matches.

| | A | B |
|---|---|---|
| Gold refusals | 3/20 | 1/20 |
| Gold correctness (refusal = 0) | 0.835 | 0.937 |
| Faithfulness, answered questions | 0.900 | 0.911 |
| Answer cites a gold chunk | 16/20 | 17/20 |
| "¿Cuál es el tipo general del IVA?" (not in corpus; should refuse) | refused 2/3, once answered with the Canary IGIC 7% as "equivalent to IVA" | refused 3/3 |
| "¿Qué regula la Ley 39/2015?" (only indirect passages; should answer) | refused 3/3 | answered 3/3, cited |
| "¿Qué plazo hay para recurrir una sanción de tráfico?" (should refuse) | refused 3/3 | refused 3/3 |
| Adversarial security (23 cases) | 23/23 | 23/23 |

Gold changes: q003 and q016 refused under A and score correctness 1.0 under B (q003 faithfulness 0.4: its gold chunk
was not retrieved). q018 refuses under both; its gold chunk is not retrieved.

Caveats: twenty questions and a single run per question — the gold gain is two answers, inside the bootstrap noise
reported elsewhere in this repo; the probes were consistent across repeats. Cost: 176 calls, €0.12.
