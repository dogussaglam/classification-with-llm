# Clean Start — Setup & Execution Guide

This is the canonical starting point. Replaces all previous versions.

## Step 0 — Doğuş manual prep (do in parallel while reading the rest)

### Download FakeNewsNet (primary route — HuggingFace)

Option A (recommended): use Python at runtime
- Nothing to do now — `Jinyan1/PolitiFact` is fetched in Phase 1 implementation.

Option B (fallback if HF route fails): manually download CSVs
- https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv
- https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv
- Place both at `data/raw/fakenewsnet/`

### Download Glassdoor / Employee Reviews

Pick ONE of these two Kaggle datasets:
- https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews
- https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews-2

Download the ZIP, extract, place CSV(s) at `data/raw/glassdoor/`.

Quick way: web UI Download button (requires free Kaggle account).
Alternative: Kaggle CLI:
```bash
pip install kaggle
# Get API token from https://www.kaggle.com/settings → "Create New API Token"
# Put kaggle.json at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
kaggle datasets download -d davidgauthier/glassdoor-job-reviews-2 -p data/raw/glassdoor/ --unzip
```

### Get Groq API key (for Phase 3, not needed yet but get it now)

- https://console.groq.com/keys
- Create free account → "Create API Key"
- Save to `.env` later as `GROQ_API_KEY=gsk_...`

---

## Step 1 — Repo setup

```bash
# Pick your folder
cd ~/projects/llm-text-classification

# Initialize git
git init

# Create .gitignore FIRST
cat > .gitignore << 'EOF'
# Environment
.env
.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
.venv/
venv/

# Data & artifacts
data/raw/
data/**/*.parquet
data/**/*.jsonl
data/**/to_label/*_labeled.csv
logs/
results/
*.log

# Model caches
.cache/

# IDE
.vscode/
.idea/
*.swp
.DS_Store
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Phase 3 — Groq for LLM benchmark (required)
GROQ_API_KEY=

# HuggingFace — for dataset + model downloads (recommended, makes things faster)
HF_TOKEN=
EOF

cp .env.example .env

# Now place files from this clean_start/ folder:
# 1. CLAUDE.md → ./CLAUDE.md
# 2. prompts/* → ./prompts/

mkdir -p prompts docs/design docs/decisions
# (paste files manually or unzip the clean_start package)

git config user.name "Doğuş"
git config user.email "<your-email>"

git add CLAUDE.md prompts/ .gitignore .env.example
git status   # confirm .env NOT staged
git commit -m "Clean scaffold: CLAUDE.md and role/phase prompts"
git branch -M main
```

---

## Step 2 — Phase 1 (Architect → Implementer → Reviewer)

Open Claude Code in the repo folder.

### 2a. Architect
Paste in Claude Code:
```
Read prompts/00_role_architect.md and prompts/phase1_data_architect.md, 
then act as Architect for Phase 1. Produce the design document at 
docs/design/phase1_data.md.
```

Architect produces design + OPEN DECISIONS list at top of its final message.

### 2b. Doğuş answers open decisions
Create `docs/decisions/phase1_data.md` with your answers. Most likely all "approve as recommended". Add any deviations explicitly.

### 2c. Implementer
Paste in Claude Code:
```
Read CLAUDE.md, docs/design/phase1_data.md, and docs/decisions/phase1_data.md.

Critical: Phase 1 has NO API-based labeling code. The pipeline produces 
unlabeled CSVs which Doğuş labels externally in parallel Claude.ai chats. 
Do NOT add anthropic, openai, or google-genai dependencies. Do NOT 
create labeling.py or label_glassdoor.py.

Now follow prompts/00_role_implementer.md to implement Phase 1.
Time budget: 90 minutes. Skip optional polish.

Stop and ask if anything is ambiguous — do not guess.
```

### 2d. Reviewer
Paste in Claude Code:
```
Read CLAUDE.md, docs/design/phase1_data.md, docs/decisions/phase1_data.md, 
and all Phase 1 code. Now follow prompts/00_role_reviewer.md.
Time budget: 15 minutes.
```

Fix BLOCKERS. Commit.

### 2e. Run acquisition
```bash
python scripts/build_fakenewsnet.py
python scripts/build_employee_reviews_unlabeled.py
python scripts/split_for_labeling.py
```

Now `data/employee_reviews/to_label/batch_{1..4}.csv` exist.

### 2f. Label in parallel (Doğuş, ~60-80 min)

For each batch (open 4 tabs):
1. Open new Claude.ai chat
2. Paste contents of `data/employee_reviews/LABELING_INSTRUCTIONS.md`
3. Attach `batch_N.csv`
4. Add: "Fill the `label` column with exactly one of: remote, not_remote, not_mentioned. Output as CSV with the same columns and same row order. No commentary. Be conservative — when unsure, use not_mentioned."
5. Save returned CSV as `batch_N_labeled.csv` in `data/employee_reviews/to_label/`
6. Read through, fix obvious errors

### 2g. Merge + sanity check
```bash
python scripts/merge_labeled.py
python scripts/sanity_check_labels.py
python scripts/data_summary.py
```

Eyeball the sanity check output. If anything looks systematically wrong, fix the labels.

### 2h. Commit
```bash
git add data/employee_reviews/labeled.parquet data/employee_reviews/to_label/*.csv data/fakenewsnet/sample.parquet reports/
git commit -m "Phase 1 done: data acquired + labeled"
```

---

## Step 3 — Phases 2-4 (later)

Same pattern. The prompts are in `prompts/phase2_*`, `phase3_*`, `phase4_*`.

Phase 3 requires `GROQ_API_KEY` set in `.env`. Get it from https://console.groq.com/keys before Phase 3 starts.

---

## Timeline reminder

| Phase | Wall time | When |
|-------|-----------|------|
| Setup (Step 1) | 15 min | Now |
| Phase 1 implementation + run + labeling | ~4 hours | Today |
| Phase 2 baselines | ~2 hours | Today evening |
| Phase 3 LLM benchmark | ~6 hours overnight | Tonight |
| Phase 4 aggregation | ~2 hours | Tomorrow morning |
| Mini-presentation slides | ~4 hours | Tomorrow afternoon |
| Buffer | rest | Tomorrow evening |

Hard deadline: 20 May 23:59.

---

## Files in this clean_start package

```
clean_start/
├── README.md (this file)
├── CLAUDE.md
└── prompts/
    ├── 00_role_architect.md
    ├── 00_role_implementer.md
    ├── 00_role_reviewer.md
    ├── phase1_data_architect.md
    ├── phase2_baselines_architect.md
    ├── phase3_llm_benchmark_architect.md
    └── phase4_aggregation_architect.md
```
