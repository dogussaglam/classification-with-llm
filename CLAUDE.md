# LLM Text Classification — Replication & Extension Study

## Project goal

Replicate Kostina et al. 2025 (arXiv:2501.08457) "Large Language Models for Text Classification: A Case Study and Comprehensive Review" on our own data sample, then extend along three contributions:

1. **Newer model generation** — benchmark 2025 LLMs absent from the original paper
2. **Reasoning-augmented inference** — does built-in reasoning replace manual Chain-of-Thought?
3. **Enriched baselines** — add Random Forest and XGBoost (instructor feedback)

## Hard deadline

**20 May 2026, 23:59** — project complete, code + results + mini-presentation ready.

Current time: 19 May 2026, ~15:00. Total budget: ~33 hours including sleep.

Mini-presentation delivery: 21 May. Full report: 5 June (separate scope).

## Datasets

### FakeNewsNet (binary)
- Source: `Jinyan1/PolitiFact` on HuggingFace (primary), fallback: KaiDMML/FakeNewsNet GitHub politifact CSVs
- Target: 214 records, 50/50 balance, articles ≤4096 tokens
- Sampling: deterministic with `seed=42`
- Doğuş downloads raw data manually and places in `data/raw/fakenewsnet/`

### Employee Reviews (3-class)
- Source: Public Glassdoor reviews dataset on Kaggle (verified at runtime)
- Target: 200 records, ~33% per class (`remote` / `not_remote` / `not_mentioned`), each ≤4096 tokens
- Doğuş downloads raw data manually and places in `data/raw/glassdoor/`

## Labeling strategy

**Pure human labeling** with parallel-chat assist for speed:
- Pipeline produces `unlabeled.parquet` → splits into 4 CSV files of 50 rows each
- Doğuş opens 4 parallel Claude.ai chats, uploads one CSV per chat, gets labels back
- Doğuş REVIEWS every label and fixes errors (human-in-the-loop)
- Pipeline merges 4 labeled CSVs into final `labeled.parquet`

The output is human-validated. No API spend, no third-party labeling model accountability.

## Models

### Baselines (5)
- Naive Bayes + TF-IDF
- SVM + TF-IDF (LinearSVC)
- Random Forest + TF-IDF (instructor request)
- XGBoost + TF-IDF (instructor request)
- RoBERTa-base, fine-tuned (HuggingFace)

### LLM benchmark (4, all via Groq free tier)
- Llama 3.3 70B (Axis A — newest large)
- Qwen3 8B (Axis A — newer small, different family)
- Gemma 2 9B (paper-era replication anchor)
- DeepSeek-R1-Distill-Llama-8B or -70B (Axis B — reasoning)

Slugs verified at Phase 3 runtime against Groq's available models.

## Prompting strategies (3)

1. `ZS` — Zero-Shot baseline
2. `ZS_CoT` — Zero-Shot Chain-of-Thought
3. `FS_CoT_RP_NA` — Few-Shot + CoT + Role-Play + Naming the Assistant (paper's best on harder task)

Few-shot examples drawn from a held-out subset (last 4-6 rows after seeded shuffle), NEVER in test set.

## Evaluation

- Primary metric: **weighted F1 score**
- Secondary: total inference latency (Pareto axis)
- Baselines: 5-fold stratified CV
- LLMs: single deterministic pass at `temperature=0`

## Validation gate (after Phase 2)

Our baselines must land within ±5 F1 of the paper's numbers:

| Model | FakeNewsNet target | Employee Reviews target |
|-------|-------------------|------------------------|
| Naive Bayes | 90.0 | 61.3 |
| SVM | 88.8 | 68.7 |
| RoBERTa | 93.0 | 83.8 |

±10 = soft warning. Outside ±10 = stop and debug before Phase 3.

## Workflow

Multi-agent: **Architect → Implementer → Reviewer** with gate approvals.

- Phase 1: data acquisition + (pipeline only, no labeling code)
- Phase 2: baselines (NB, SVM, RF, XGBoost, RoBERTa)
- Phase 3: LLM benchmark (4 models × 3 prompts × 414 records ≈ 5,000 Groq calls)
- Phase 4: aggregation + figures + report

Role prompt templates live in `prompts/`. Phase-specific prompts also live in `prompts/`.

## Code conventions

- Python 3.11+
- Type hints on all public functions
- Docstrings on all public functions: one-line summary + Args + Returns
- pytest for non-trivial logic only
- Format: `ruff format` + `ruff check`
- All experiments deterministic: fix all random seeds
- Results: Parquet for tabular, JSON/JSONL for raw model outputs, Markdown for human-readable summaries
- Logs: `logs/<phase>-<timestamp>.log`, not stdout (except progress bars)
- Dependencies: `pyproject.toml`

## Environment variables (used, never committed)

```
GROQ_API_KEY            # Phase 3 (required)
HF_TOKEN                # HuggingFace dataset/model download (recommended)
# Anthropic, OpenAI, Google keys NOT used in this pipeline
```

Stored in `.env` (gitignored). `.env.example` lists keys with empty values.

## What NOT to do

- Do not use any LLM API for labeling — human-in-the-loop is the design
- Do not test on data used as Few-Shot examples (leakage)
- Do not skip the Phase 2 baseline validation gate
- Do not commit API keys, raw scraped data, or anything in `data/raw/`
- Do not invent abstractions not in the approved design — stop and re-architect if needed
- Do not run Phase 3 without explicit Doğuş approval after Phase 2 gate passes

## Repository layout (target)

```
.
├── CLAUDE.md                    # this file
├── README.md                    # written in Phase 4
├── pyproject.toml
├── .env.example
├── .env                         # gitignored
├── .gitignore
├── prompts/                     # role + phase prompts
├── docs/
│   ├── design/                  # one .md per phase
│   └── decisions/               # Doğuş's answers to architect open decisions
├── src/llm_textcls/
│   ├── data/                    # acquisition only — no labeling
│   ├── baselines/               # NB, SVM, RF, XGBoost, RoBERTa
│   ├── llms/                    # Groq client, prompts, runner
│   ├── evaluation/              # F1, timing
│   └── reporting/               # tables, figures
├── tests/
├── scripts/                     # CLI entry points
├── data/
│   ├── raw/                     # Doğuş-downloaded source (gitignored)
│   ├── fakenewsnet/             # processed
│   └── employee_reviews/        # processed + labeled
├── results/
│   ├── baselines/
│   ├── llms/
│   └── master/
├── reports/                     # human-readable outputs
└── logs/
```
