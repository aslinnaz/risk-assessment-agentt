# TrustAI Risk Assessment — prototype

**Language note:** the application's user interface, interview questions, options, rationales, LLM instructions and generated reports are in **Turkish**. This README, code comments and docstrings are in English.

## Purpose
The application chats with a **human** who knows the AI agent being assessed and produces a risk *preliminary assessment* based on **user declarations** (TrustAI labels: LOW / MEDIUM / HIGH / FORBIDDEN, or "assessment could not be completed").
Legal applicability is a separate field and defaults to "not determined". The operational assessment is based on the TrustAI playbooks and the NIST AI RMF Playbook.

Stack: Python, Streamlit, OpenAI API (chat + embeddings), local BM25 plus a persistent embedding index.

## Installation
```bash
git clone <repo-url> && cd <repo-folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # put your OPENAI_API_KEY in .env 
```
The chat model and embedding model are selected through environment variables: `OPENAI_MODEL` (default `gpt-4.1-mini`) and `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`). Streamlit secrets are also supported.

## Preparing the source files
The source PDFs are **not included** in the repository (distribution permission has not been verified). Put your own copies in `sources/` **using these names**:

| File name | Source | Note |
|---|---|---|
| `EU_AI_Act_OJ_L_2024_1689.pdf` | Regulation (EU) 2024/1689, OJ L, 12.7.2024 — EUR-Lex | Must be the original 2024 text |
| `EU_Commission_Decision.pdf` | Commission Implementing Decision (EU) 2026/2274 — Official Journal | Only the harmonised-standard reference |
| `TrustAI_Agentic_Governance_Playbook.pdf` | TrustAI (2026) | From the TrustAI team |
| `TrustAI_Governance_Playbook.pdf` | TrustAI (v1.0, 2026) | From the TrustAI team |
| `AI_RMF_Playbook.pdf` | NIST AI RMF Playbook — airc.nist.gov | |
| `ISO_42001_2023.pdf` | ISO/IEC 42001:2023 |

Version and date information is in `catalogs/sources.json`. If the files are missing, the application says so and disables the start button. Source texts are treated as data; instructions inside them are not applied.

On first run the passages are extracted locally and written to `.cache/`. The embedding index for semantic search is built **only by an explicit user action** (sidebar → Setup → *Embedding indeksini oluştur*, "Build embedding index"); it has a small one-off API cost. If it is not built, search uses local keyword matching only.

## Running
```bash
streamlit run app.py
```
In the sidebar, "Değerlendirme ayarları" (assessment settings) lets you change the agent's operating scope (default Türkiye) and its EU link (default *Belirlenmedi*, "not determined"). User answers and reports are kept in the session only and are never written to the repository.

Local checks (the source PDFs must be in `sources/`; no API calls are made): `python -m pytest -q` and `python tools/verify_catalog.py`.

## Structure
`app.py` UI · `interview.py` interview (one model call per answer) · `planner.py` question selection and finishing (no model) · `classify.py` state derivation and classification · `rag.py` extraction / index / retrieval · `report.py` report and PDF · `catalogs/` source manifest, C01–C21 coverage map, interview topics and decision table.

## Interview summary
The target is roughly 10–15 questions, with a hard limit of 20 including follow-ups. Question selection happens in `planner.py` (no model call); the model only extracts declarations and phrases the question among the candidate topics. The report's citation check is programmatic; the additional model-based support check is off by default.

## Current limitations (not complete)
- **Not sufficiently tested with a real model.** The local tests use a scripted fake client; question quality, slot-extraction accuracy and rationale fidelity have not been verified with a real model.
- **Question flow** (repetition prevention, follow-up, conflict and finishing rules) and the data-consistency fixes were recently reworked and have not been sufficiently validated in real interviews. Repeated or unnecessary questions, or wrongly recorded fields, may occur.
- **The semantic accuracy of source mappings has not been verified.** Citations are checked only programmatically (passage existence + keyword). The semantic support check is an optional single model call and does not replace human review.

- C01–C21 is a coverage map; most controls are not asked and remain "not yet assessed". Coverage is not a safety score.
- The target agent is not technically tested; answers are not verified.
- No license has been added yet; the code and catalogs are for review before sharing.
