# Bidirectional RAG (KnowledgeOS)

Implementation of *"Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation Through Multi-Stage Validation"* — only what the paper describes: forward path, acceptance layer (grounding → attribution → novelty), write-back, experience store, the three compared systems, the experimental protocol, and Tables I/II + Figures 1/2.

## Setup
```bash
pip install -r requirements.txt
ollama serve            # separate terminal
ollama pull llama3.2:3b
```

## Run
```bash
python run_experiments.py --corpus-dir data/corpus --datasets-dir data/datasets
```
Defaults = paper protocol: NQ, TriviaQA, HotpotQA, Stack Overflow × seeds 42/43/44, 500 queries (400 train / 100 test), 3 systems → 12 experiments per system. Output goes to `results/` (`results.md`, `table1_*.csv`, `table2_*.csv`, `fig1_*.png`, `fig2_*.png`, `raw_runs.json`).

Quick trial: `--datasets nq --seeds 42 --queries 50 --train 40 --test 10`

Inputs:
- `data/corpus/<dataset>/` — seed corpus per dataset (`.txt`, `.md`, or `.jsonl` with a `text` field), ingested once.
- `data/datasets/<dataset>.jsonl` (optional) — `{"question": "..."}` per line. If absent, queries are pulled from HuggingFace.

## Smoke tests
```bash
python -m unittest tests.smoke_test -v
```
These replace only the three model backends (Ollama, MiniLM, DeBERTa NLI) with tiny deterministic stand-ins; all other code is the real implementation.

## Layout
| File | Paper section |
|---|---|
| `birag/config.py` | thresholds and settings (0.65, 0.10, 0.4, seeds, 400/100) |
| `birag/models.py` | V-D: MiniLM, Ollama llama3.2:3b, DeBERTa-v3 NLI |
| `birag/corpus.py` | D_t on ChromaDB (cosine), ingestion, write-back W |
| `birag/acceptance.py` | IV-B: Eq. 8 grounding, Eq. 9 attribution, Eq. 10 novelty |
| `birag/experience.py` | IV-C: experience store |
| `birag/rag.py` | IV-A + V-B: Standard RAG, Naive Write-back, Bidirectional RAG |
| `birag/data.py`, `experiment.py`, `report.py` | V: datasets, protocol, metrics, tables, figures |

## Things the paper does not specify (my minimal choices — all configurable)
- **Seed corpus**: not described; you supply it per dataset.
- **Retrieval top-k**: 5. **Experience notes injected**: 3. **Chunk size**: 500 chars (one chunk per paragraph).
- **Attribution threshold**: no value given; default 1.0 (every citation must reference a retrieved doc; no citations = 0).
- **Citation F1**: precision = valid cited / cited; recall = valid cited / retrieved.
- **Latency** is measured on training queries (full pipeline incl. validation); **coverage and citation F1** on the test queries after training, with write-back off.
- **Experience critiques** are rule-based text from the failed check, not LLM-written.
- **Std** is population std over the 12 experiments.
- The paper reports 500 documents added by Naive Write-back but also 400 training queries; with that split Naive adds at most 400, so growth numbers will not match 500/140 exactly.
- The paper's αmax and ε_h constraints (Eq. 2–3) are not enforced (no values given); α is reported.
