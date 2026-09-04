# Summary: Polysomnography & Obstructive Sleep Apnea Knowledge Base

This knowledge base is curated from 5 recent high-quality review and meta-analysis papers for building a RAG system focused on **polysomnography (PSG)** related questions, particularly around diagnosis, treatment, pathophysiology, and adherence in obstructive sleep apnea (OSA/OSAHS).

## Documents Included

| ID | Title | Year | Type | Focus |
|----|-------|------|------|-------|
| positional_therapy | Comparative efficacy of SPT, OAT, and CPAP | 2025 | Meta-analysis | Treatment comparison (AHI, safety) |
| rem_sleep | Functional roles of REM sleep | 2023 | Review | Physiology of REM (memory, synapses) |
| vascular_lesions | OSAHS and vascular lesions | 2024 | Review | Pathophysiology & multi-organ vascular effects |
| cpap_adherence | Psychological predictors of CPAP adherence (3P model) | 2025 | Scoping review | Barriers & motivators to CPAP use |
| mad_devices | Efficacy & adherence of different MAD designs | 2025 | Meta-analysis | Titratable/custom vs other MAD designs |

## Key Themes for PSG / Sleep Medicine RAG

1. **Diagnostics & Metrics (PSG core)**  
   AHI (total/supine/non-supine), ODI, arousal index, TST, sleep efficiency, SaO2 parameters, ESS, position dependence (POSA).

2. **Treatment Modalities**  
   - CPAP (gold standard, adherence challenges)  
   - Oral appliances / MADs (custom > ready-made for adherence; titratable ≈ nontitratable for efficacy)  
   - Sleep Positional Therapy (good for POSA, safer, less effective on overall AHI than CPAP)

3. **Pathophysiology**  
   Intermittent hypoxia → endothelial dysfunction, oxidative stress, inflammation → coronary, aortic, cerebrovascular, renal, retinal, and uterine vascular lesions.

4. **Adherence Psychology**  
   3P model: Predisposing (literacy, comorbidities), Precipitating (anxiety, stigma), Perpetuating (support, intimacy). Motivators: perceived benefit, partner & clinician support, education.

5. **REM Sleep Physiology**  
   Neural replay, theta modulation, synaptic pruning/strengthening, memory consolidation — relevant for understanding sleep architecture changes seen on PSG.

## File Structure for RAG

- `metadata.json` — document-level metadata
- `clean_text.txt` — full cleaned text of all papers
- `chunks.json` — ~400-word overlapping chunks with provenance
- `questions.json` — 12 high-quality seed Q&A pairs
- `keywords.json` — domain + frequency-based terms
- `entities.json` — structured named entities (diseases, devices, metrics…)
- `rag_ready.json` — unified payload ready for embedding / vector store ingestion
- `summary.md` — this file

## Suggested RAG Usage

1. Embed `chunks.json` texts (use `chunk_id` as ID).
2. Use `metadata.json` + `entities.json` for filtering / hybrid search.
3. Seed evaluation with `questions.json`.
4. Expand keywords for query expansion / synonym handling.
