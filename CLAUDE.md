# LLM Text Classification — Replication & Extension Study

## Project goal

Replicate Kostina et al. 2025 (arXiv:2501.08457) "Large Language Models for Text Classification: A Case Study and Comprehensive Review" on our own data sample, then extend with three contributions:

1. **Newer model generation** — benchmark 2025 LLMs absent from the original paper (Llama 3.3 70B, Qwen3 32B/8B, DeepSeek-V3 distill, via Groq)
2. **Reasoning-augmented inference** — test DeepSeek-R1-Distill: does built-in reasoning replace manual Chain-of-Thought?
3. **Enriched baselines** — add Random Forest and XGBoost to the original NB/SVM/RoBERTa baselines (instructor feedback)

## Deadlines

- **21 May 2026**: mini-presentation with working code + preliminary metric tables
- **5 June 2026**: full report

## Datasets

### FakeNewsNet (binary)
- Source: https://github.com/KaiDMML/FakeNewsNet — public `dataset/politifact_fake.csv` and `dataset/politifact_real.csv`
- Target: 214 records, 50/50 balance, articles ≤4096 tokens
- Sampling: deterministic with `seed=42`
- The original CSVs contain id + URL + title + tweet_ids. Full article text must be retrieved (scrape from URL or use a cached copy if found).

### Employee Reviews (3-class)
- Source: public Glassdoor reviews dataset on Kaggle (specific dataset selected in Phase 1)
- Target: 1000 records, ~33% per class (`remote` / `not_remote` / `not_mentioned`), each ≤4096 tokens
- **Labeling: dual-LLM ensemble with manual reconciliation**
  - Two independent labelers from different model families (default: Claude Sonnet 4.5 via Anthropic API + GPT-4o-mini via OpenAI API)
  - Agreement cases → automatic label (~expected 85-90% of data)
  - Disagreement cases → manual review by Doğuş
  - **Gold-standard subset**: Doğuş manually labels 100 random examples; we compute LLM-labeler accuracy and Cohen's kappa against this gold subset

## Models

### Baselines (5 — all on same data, used to validate pipeline)
- Naive Bayes + TF-IDF
- SVM + TF-IDF
- Random Forest + TF-IDF (new — instructor request)
- XGBoost + TF-IDF (new — instructor request)
- RoBERTa-base, fine-tuned

### Original-paper LLMs to replicate (4 — for direct within-our-data comparison)
- Llama 3 70B (Groq)
- Llama 3 8B (Groq)
- Gemma 2 9B (Groq)
- Mistral 7B OpenOrca (HuggingFace local, AWQ 4-bit)

### New LLMs (Axis A — primary contribution)
- Llama 3.3 70B (Groq)
- Qwen3 32B (Groq)
- Qwen3 8B (Groq)
- DeepSeek-V3 distill (Groq, if available — verify in Phase 3)

### Reasoning model (Axis B)
- DeepSeek-R1-Distill-Llama-8B (preferred Groq, fallback HuggingFace local)

## Prompting strategies (5 selected from paper's 10)

1. `ZS` — Zero-Shot baseline
2. `ZS+CoT` — Zero-Shot + Chain-of-Thought
3. `FS` — Few-Shot (2 examples for FNN, 3 examples for ER)
4. `ZS+RP+NA` — Zero-Shot + Role-Playing + Naming the Assistant (paper's best on FNN)
5. `FS+CoT+RP+NA` — Few-Shot + CoT + RP + NA (paper's best on ER)

## Evaluation
- Primary metric: **weighted F1 score**
- Secondary metric: total inference time (Pareto axis)
- Baselines: 5-fold cross-validation
- LLMs: single deterministic pass at `temperature=0`
- Few-shot examples drawn from a held-out subset NEVER used in test

## Validation gate (critical)

After Phase 2 (baselines), run our baselines on our data sample. Compare to paper's baseline numbers. **Accept if within ±5 F1 points.** If outside this band, stop and investigate before proceeding to LLM benchmarks. Document any deviation transparently in the final report.

Paper reference numbers for validation:

| Model | FakeNewsNet | Employee Reviews |
|-------|-------------|------------------|
| Naive Bayes | 90.0 | 61.3 |
| SVM | 88.8 | 68.7 |
| RoBERTa | 93.0 | 83.8 |

(RF and XGBoost are new — no paper baseline. Report as-is.)

## Workflow

Multi-agent pattern: **Architect → Implementer → Reviewer** with gate approvals.

For each phase:
1. **Architect** prompt → produces `docs/design/<phase>.md`, lists open decisions
2. Doğuş reviews design, answers open decisions, approves
3. **Implementer** prompt → codes against the approved design
4. **Reviewer** prompt → audits implementation, flags BLOCKERS / WARNINGS / NITS
5. Doğuş fixes BLOCKERS, commits

Role prompt templates are in `prompts/` directory.

## Code conventions

- Python 3.11+
- Type hints on all public functions
- One-line docstring + Args + Returns on all public functions
- pytest for non-trivial logic (no need for trivial wrappers)
- Format: `ruff format` + `ruff check`
- All experiments deterministic: fix `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and any HF seeds
- Results: Parquet for tabular, JSON for raw model outputs, Markdown for human-readable summaries
- Logs: `logs/<phase>-<timestamp>.log`, not stdout (except progress bars)
- Dependencies: `pyproject.toml` with uv (preferred) or `requirements.txt` (acceptable)

## Environment variables (used, never committed)

```
ANTHROPIC_API_KEY      # For Claude labeling
OPENAI_API_KEY         # For GPT-4o-mini labeling
GROQ_API_KEY           # For all benchmark LLMs
HF_TOKEN               # For HuggingFace model downloads
KAGGLE_USERNAME        # For Kaggle CLI
KAGGLE_KEY             # For Kaggle CLI
```

Store in `.env` (gitignored). Provide `.env.example` with empty values.

## What NOT to do

- Do not use any benchmark LLM as a labeling LLM (circular benchmarking)
- Do not test on data that was used as Few-Shot examples (leakage)
- Do not skip the baseline-validation gate before running LLM benchmarks
- Do not commit API keys, raw scraped data with copyright concerns, or anything in `data/raw/`
- Do not invent abstractions not in the approved design — if you find you need one, stop and re-architect

## Repository layout (target)

```
.
├── CLAUDE.md                    # this file
├── README.md                    # human-facing, written in Phase 5
├── pyproject.toml
├── .env.example
├── .gitignore
├── prompts/                     # role + phase prompts
├── docs/
│   ├── design/                  # one .md per phase
│   └── decisions/               # open decisions answered by Doğuş
├── src/
│   └── llm_textcls/
│       ├── data/                # acquisition + labeling
│       ├── baselines/           # NB, SVM, RF, XGBoost, RoBERTa
│       ├── llms/                # Groq clients, HF local clients, prompt templates
│       ├── evaluation/          # F1, timing, Pareto utilities
│       └── reporting/           # tables, figures, summaries
├── tests/
├── scripts/                     # one-off CLI entry points
├── data/
│   ├── raw/                     # downloaded source (gitignored)
│   ├── fakenewsnet/             # processed
│   └── employee_reviews/        # processed
├── results/
│   ├── baselines/
│   └── llms/
├── reports/                     # human-readable outputs
└── logs/
```
