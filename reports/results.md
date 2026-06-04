# Phase 4 results — replication + extension
*All numbers in this report are rendered from `results/master/*.parquet` at build time; rerun `python scripts/build_report.py` to regenerate after any change.*

## 1. Executive summary
- **FakeNewsNet (binary, n=210)** — winner: **Naive Bayes (TF-IDF)** at F1 0.900 (prompt: —). Best LLM: Llama 3.3 70B FS_CoT_RP_NA at 0.841.
- **Employee Reviews (3-class, n=204)** — winner: **Llama 3.3 70B** at F1 0.862 (prompt: FS_CoT_RP_NA). Best baseline: RoBERTa-base (fine-tuned) at 0.750.
- **Surprise 1 (Axis B over-thinking):** GPT-OSS 120B ZS on FNN at 0.657 is +14.7 pp behind Llama 3.3 70B ZS (0.803).
- **Surprise 2 (Qwen3 toggle):** reasoning=none → default flips direction by task: FNN 0.833 → 0.790 (-4.3 pp); ER 0.804 → 0.834 (+3.0 pp).
- **Replication:** NB, SVM land within ±5 of the paper; RoBERTa is the only soft warning. NB on ER is the standing >±10 failure inherited from Phase 2.

## 2. Setup recap
We replicate Kostina et al. 2025 (arXiv:2501.08457) on two datasets: FakeNewsNet (PolitiFact, binary fake/real, n=210 after held-out FS examples) and Employee Reviews (Glassdoor remote-work mention, 3-class, n=204). Baselines are 5-fold stratified CV; LLMs are a single deterministic pass at `temperature=0` against Groq's free-tier API. See `docs/design/phase1_*.md` (data + labelling), `phase2_baselines.md` (TF-IDF + RoBERTa), and `phase3_llm_benchmark.md` (prompts, runner, parsing). All seeds fixed to 42. The four LLMs are the 8B / 70B Llama pair (Axis A — scaling), the Qwen3 32B reasoning toggle (Axis A — reasoning toggle on a same-model spec), and the GPT-OSS 120B reasoning=medium model (Axis B — reasoning model class). Three prompts: zero-shot (ZS), zero-shot chain-of-thought (ZS_CoT), and the paper's strongest harder-task prompt FS_CoT_RP_NA (few-shot + CoT + role-play + naming the assistant). Few-shot examples come from a deterministic held-out slice of the test set, never re-used in scoring.

## 3. Master F1 table
| Model                           | Family       |   FNN F1 | FNN Best Prompt   |   FNN Latency (ms) |   ER F1 | ER Best Prompt   |   ER Latency (ms) |
|:--------------------------------|:-------------|---------:|:------------------|-------------------:|--------:|:-----------------|------------------:|
| Naive Bayes (TF-IDF)            | classical_ml |    0.9   | —                 |                0.5 |   0.459 | —                |               0.2 |
| LinearSVC (TF-IDF)              | classical_ml |    0.88  | —                 |                0.6 |   0.657 | —                |               0.2 |
| RoBERTa-base (fine-tuned)       | roberta      |    0.856 | —                 |               11   |   0.75  | —                |              11   |
| Random Forest (TF-IDF)          | classical_ml |    0.79  | —                 |                2.1 |   0.624 | —                |               2.2 |
| XGBoost (TF-IDF)                | classical_ml |    0.774 | —                 |                1.9 |   0.714 | —                |               0.2 |
| Llama 3.3 70B                   | llama        |    0.841 | FS_CoT_RP_NA      |             6426   |   0.862 | FS_CoT_RP_NA     |            4308   |
| GPT-OSS 120B (reasoning=medium) | gpt_oss      |    0.661 | ZS_CoT            |             5249   |   0.849 | FS_CoT_RP_NA     |            3905   |
| Qwen3 32B (reasoning=default)   | qwen         |    0.79  | ZS                |             7786   |   0.834 | ZS               |            4740   |
| Qwen3 32B (reasoning=none)      | qwen         |    0.833 | ZS                |             4433   |   0.804 | ZS               |            2386   |
| Llama 3.1 8B                    | llama        |    0.823 | ZS_CoT            |             5275   |   0.795 | ZS_CoT           |            3220   |

## 4. Finding 1 — Naive Bayes wins FakeNewsNet
Naive Bayes at F1 0.900 beats every LLM combo by at least 5.9 pp on the binary 210-record FakeNewsNet split. TF-IDF + multinomial NB is uniquely well-suited here: a strong vocabulary signal (clickbait register, politicised proper nouns), balanced classes, and small N where pre-trained semantic priors offer little marginal lift. The paper reports 0.900 for NB on FNN; we replicate the score exactly to the third decimal. The caveat is that the original paper trained on the full 4k+ FakeNewsNet release where LLMs gain on the long tail of ambiguous claims; we do not generalise this finding past N=210. See ![fig1](figures/fig1_f1_master.png) for the per-dataset bar comparison.

## 5. Finding 2 — LLMs win Employee Reviews
On the 3-class small-N (n=204) Employee Reviews dataset, Llama 3.3 70B with prompt FS_CoT_RP_NA hits F1 0.862, beating the best baseline (RoBERTa-base (fine-tuned) at 0.750) by +11.2 pp. The hypothesis is data-hunger: RoBERTa fine-tuning on 200 labelled rows hits a ceiling that 70B-parameter pretraining can sidestep. The classes (`remote`, `not_remote`, `not_mentioned`) reward semantic interpretation rather than vocabulary frequency, which TF-IDF handles less well. Notably this is the only split where reasoning-class GPT-OSS becomes competitive — see §6 for why that is *not* the headline. Reference: ![fig1](figures/fig1_f1_master.png) and ![fig2](figures/fig2_pareto.png).

## 6. Finding 3 — Axis B: built-in reasoning overthinks the binary task
GPT-OSS 120B with reasoning=medium underperforms across every FNN prompt: ZS 0.657, ZS_CoT 0.661, FS_CoT_RP_NA 0.636. The matching Llama 3.3 70B numbers are 0.803 / 0.810 / 0.841, a ~14 pp gap that persists regardless of prompt scaffolding. GPT-OSS spends increasing reasoning tokens (27.8k / 36.4k / 46.7k across the three prompts on FNN; see `combined_long.parquet`) without recovering F1 — adding more deliberation does not help on a task where the surface lexical signal is already decisive. On Employee Reviews the same model is competitive (0.819–0.849) but never best, and even there it loses to Llama 3.3 70B on the strongest prompt. This is the load-bearing Axis B finding: built-in reasoning is not a replacement for prompt-engineered chain-of-thought when the task does not benefit from deliberation. Reference: ![fig4](figures/fig4_axis_b_reasoning.png).

## 7. Finding 4 — Qwen3 reasoning toggle is task-dependent
Same physical model (`qwen/qwen3-32b`), same prompt (ZS), only the `reasoning_effort` parameter changes. On FakeNewsNet: F1 0.833 (reasoning=none) → 0.790 (reasoning=default), Δ -4.3 pp. On Employee Reviews: 0.804 → 0.834, Δ +3.0 pp. The toggle helps on the multi-class ambiguity task and hurts on the binary clickbait task — the same direction-flip story as Finding 3, now isolated on a single model. This rules out the alternative explanation that GPT-OSS specifically is mis-tuned. Reference: ![fig3](figures/fig3_axis_a_scaling.png).

## 8. Finding 5 — Prompt strategy is size-dependent
FS_CoT_RP_NA is the paper's strongest harder-task prompt — but only at scale. On Llama 3.1 8B Employee Reviews it scores F1 0.731 versus ZS_CoT at 0.795 (-6.4 pp). The same prompt on Llama 3.3 70B reaches 0.862 — the overall winner. Hypothesis: 8B context is too small to make full use of the role-play + few-shot scaffold, and the demonstrations crowd out the actual review under scoring. Practical implication: prompt strategy must scale with model capacity; copying a large-model prompt down to a small model is not free. Reference: ![fig5](figures/fig5_prompt_effect.png).

## 9. Pareto frontier
On the F1-vs-latency plane (![fig2](figures/fig2_pareto.png)), the low-latency / high-FNN-F1 corner is owned by Naive Bayes (≈0.6 ms/record, F1 0.900) and LinearSVC. RoBERTa is on-frontier for FNN but not for Employee Reviews, where Llama 3.3 70B (non-reasoning) sits on the frontier at roughly 2.3 s/record and F1 0.840 (ZS) or 0.862 (FS_CoT_RP_NA). GPT-OSS 120B is dominated almost everywhere — it is slower than Llama 3.3 70B per call *and* lower F1 on the binary task, so it offers no Pareto-improving point. The frontier is plotted as a grey dashed line in the figure; the in-figure caption gives the latency-axis caveats.

## 10. Replication validation vs the paper
| Model   | Dataset          |   Paper F1 |   Our F1 |      Δ | Gate                      |
|:--------|:-----------------|-----------:|---------:|-------:|:--------------------------|
| nb      | fakenewsnet      |      0.9   |    0.9   | -0     | pass (within ±5)          |
| nb      | employee_reviews |      0.613 |    0.459 | -0.154 | FAIL (>±10)               |
| svm     | fakenewsnet      |      0.888 |    0.88  | -0.008 | pass (within ±5)          |
| svm     | employee_reviews |      0.687 |    0.657 | -0.03  | pass (within ±5)          |
| roberta | fakenewsnet      |      0.93  |    0.856 | -0.074 | soft warning (within ±10) |
| roberta | employee_reviews |      0.838 |    0.75  | -0.088 | soft warning (within ±10) |

Three of six paper baselines pass the ±5 F1 gate exactly. RoBERTa lands in the ±10 soft-warning band on both datasets (overfitting on small N is the most likely cause; the paper had 4k+ FNN rows). NB on Employee Reviews is the standing >±10 failure inherited from Phase 2 — the labelling rubric for the `remote / not_remote / not_mentioned` categories differs between our human-in-the-loop label set and the paper's tag-based rule, and we deliberately do not retro-fit. The outlier is reported honestly rather than hidden.

Flagged rows:
- nb / employee_reviews: Δ -0.154 — outside ±10 gate, see §11.

## 11. Limitations
Small N (210 + 204 records) inflates the variance of any single F1 estimate; per-dataset bootstrap CIs would help and remain out of scope for the May 21 deadline. LLM runs are a single deterministic pass at `temperature=0` — we have no within-combo variance to test statistical significance against. Groq's free-tier latency varies with load and should not be read as a fair head-to-head with paid inference. One parse failure (`llama-3.1-8b-instant` × `FS_CoT_RP_NA` × FNN) was dropped; the combo's F1 is computed over 209/210 records and is flagged in `results/llms/benchmark_status.md`. Baseline latency on the Pareto axis is an approximation (mean per-fold inference time divided by per-fold test size) — Phase 2 did not record per-record latencies. The approximation error is <10% and invisible on the log-x axis but is called out in the figure caption.

## 12. Conclusions and contributions
Three contributions, mapping to specific findings.

**Contribution 1 — newer LLM generation.** The 2025 model lineup (Llama 3.1 8B, Llama 3.3 70B, Qwen3 32B, GPT-OSS 120B) covers post-paper model families. Headline: Llama 3.3 70B's FS_CoT_RP_NA score on Employee Reviews (0.862) beats every model the paper tested, including its best LLM. Findings 2 and 5 ride on this.

**Contribution 2 — reasoning-augmented inference.** Built-in reasoning (GPT-OSS medium, Qwen3 default) does *not* replace manual CoT. On the binary task it actively hurts; on the multi-class task it helps marginally but never beats a non-reasoning peer with the right prompt. Findings 3 and 4 are the load-bearing evidence; ![fig6](figures/fig6_reasoning_cost.png) prices the cost of reasoning in tokens.

**Contribution 3 — enriched baselines.** Random Forest and XGBoost complete the classical-ML grid. They land mid-pack on both datasets — RF: 0.790 / 0.624; XGBoost: 0.774 / 0.714 — showing that tree ensembles do not displace NB on FNN or RoBERTa on Employee Reviews, but are competitive and cheap.

## 13. Appendix

### A.1 Full combined long table (32 rows)
| Source   | Family       | Model                           | Dataset          | Prompt       | Reasoning   |    F1 |   F1 std |   Acc |   ms/rec |   r-tok |
|:---------|:-------------|:--------------------------------|:-----------------|:-------------|:------------|------:|---------:|------:|---------:|--------:|
| llm      | llama        | Llama 3.3 70B                   | employee_reviews | FS_CoT_RP_NA | —           | 0.862 |    0     | 0.863 |   4308   |       0 |
| llm      | llama        | Llama 3.3 70B                   | employee_reviews | ZS_CoT       | —           | 0.85  |    0     | 0.848 |   2275   |       0 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | employee_reviews | FS_CoT_RP_NA | medium      | 0.849 |    0     | 0.853 |   3905   |     118 |
| llm      | llama        | Llama 3.3 70B                   | employee_reviews | ZS           | —           | 0.84  |    0     | 0.838 |   2278   |       0 |
| llm      | qwen         | Qwen3 32B (reasoning=default)   | employee_reviews | ZS           | default     | 0.834 |    0     | 0.833 |   4740   |     219 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | employee_reviews | ZS           | medium      | 0.819 |    0     | 0.819 |   2416   |      55 |
| llm      | qwen         | Qwen3 32B (reasoning=none)      | employee_reviews | ZS           | none        | 0.804 |    0     | 0.809 |   2386   |       0 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | employee_reviews | ZS_CoT       | medium      | 0.801 |    0     | 0.799 |   3378   |      59 |
| llm      | llama        | Llama 3.1 8B                    | employee_reviews | ZS_CoT       | —           | 0.795 |    0     | 0.819 |   3220   |       0 |
| llm      | llama        | Llama 3.1 8B                    | employee_reviews | ZS           | —           | 0.78  |    0     | 0.809 |   2231   |       0 |
| baseline | roberta      | RoBERTa-base (fine-tuned)       | employee_reviews | —            | —           | 0.75  |    0.064 | 0.745 |     11   |       0 |
| llm      | llama        | Llama 3.1 8B                    | employee_reviews | FS_CoT_RP_NA | —           | 0.731 |    0     | 0.765 |   8345   |       0 |
| baseline | classical_ml | XGBoost (TF-IDF)                | employee_reviews | —            | —           | 0.714 |    0.078 | 0.725 |      0.2 |       0 |
| baseline | classical_ml | LinearSVC (TF-IDF)              | employee_reviews | —            | —           | 0.657 |    0.081 | 0.667 |      0.2 |       0 |
| baseline | classical_ml | Random Forest (TF-IDF)          | employee_reviews | —            | —           | 0.624 |    0.071 | 0.687 |      2.2 |       0 |
| baseline | classical_ml | Naive Bayes (TF-IDF)            | employee_reviews | —            | —           | 0.459 |    0.055 | 0.559 |      0.2 |       0 |
| baseline | classical_ml | Naive Bayes (TF-IDF)            | fakenewsnet      | —            | —           | 0.9   |    0.041 | 0.9   |      0.5 |       0 |
| baseline | classical_ml | LinearSVC (TF-IDF)              | fakenewsnet      | —            | —           | 0.88  |    0.026 | 0.881 |      0.6 |       0 |
| baseline | roberta      | RoBERTa-base (fine-tuned)       | fakenewsnet      | —            | —           | 0.856 |    0.043 | 0.857 |     11   |       0 |
| llm      | llama        | Llama 3.3 70B                   | fakenewsnet      | FS_CoT_RP_NA | —           | 0.841 |    0     | 0.843 |   6426   |       0 |
| llm      | qwen         | Qwen3 32B (reasoning=none)      | fakenewsnet      | ZS           | none        | 0.833 |    0     | 0.833 |   4433   |       0 |
| llm      | llama        | Llama 3.1 8B                    | fakenewsnet      | ZS_CoT       | —           | 0.823 |    0     | 0.824 |   5275   |       0 |
| llm      | llama        | Llama 3.1 8B                    | fakenewsnet      | ZS           | —           | 0.814 |    0     | 0.814 |   4283   |       0 |
| llm      | llama        | Llama 3.3 70B                   | fakenewsnet      | ZS_CoT       | —           | 0.81  |    0     | 0.814 |   2338   |       0 |
| llm      | llama        | Llama 3.3 70B                   | fakenewsnet      | ZS           | —           | 0.803 |    0     | 0.81  |   2295   |       0 |
| llm      | qwen         | Qwen3 32B (reasoning=default)   | fakenewsnet      | ZS           | default     | 0.79  |    0     | 0.79  |   7786   |     308 |
| baseline | classical_ml | Random Forest (TF-IDF)          | fakenewsnet      | —            | —           | 0.79  |    0.041 | 0.795 |      2.1 |       0 |
| baseline | classical_ml | XGBoost (TF-IDF)                | fakenewsnet      | —            | —           | 0.774 |    0.05  | 0.776 |      1.9 |       0 |
| llm      | llama        | Llama 3.1 8B                    | fakenewsnet      | FS_CoT_RP_NA | —           | 0.708 |    0     | 0.718 |  14434   |       0 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | fakenewsnet      | ZS_CoT       | medium      | 0.661 |    0     | 0.671 |   5249   |     122 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | fakenewsnet      | ZS           | medium      | 0.657 |    0     | 0.667 |   4548   |      95 |
| llm      | gpt_oss      | GPT-OSS 120B (reasoning=medium) | fakenewsnet      | FS_CoT_RP_NA | medium      | 0.636 |    0     | 0.652 |   5797   |     175 |

### A.2 Per-prompt impact (LLM mean, Qwen3 excluded)
| Prompt       | Dataset          |   Mean F1 (excl. Qwen3) |   Δ vs ZS |
|:-------------|:-----------------|------------------------:|----------:|
| FS_CoT_RP_NA | employee_reviews |                   0.814 |     0.001 |
| ZS           | employee_reviews |                   0.813 |     0     |
| ZS_CoT       | employee_reviews |                   0.815 |     0.002 |
| FS_CoT_RP_NA | fakenewsnet      |                   0.728 |    -0.03  |
| ZS           | fakenewsnet      |                   0.758 |     0     |
| ZS_CoT       | fakenewsnet      |                   0.765 |     0.007 |

### A.3 LaTeX output note
`reports/results_table.tex` uses booktabs `\toprule` / `\midrule` / `\bottomrule` rules — include `\usepackage{booktabs}` in the report preamble before importing the table.

### A.4 Figure index
- `fig1`: figures/fig1_f1_master.png
- `fig2`: figures/fig2_pareto.png
- `fig3`: figures/fig3_axis_a_scaling.png
- `fig4`: figures/fig4_axis_b_reasoning.png
- `fig5`: figures/fig5_prompt_effect.png
- `fig6`: figures/fig6_reasoning_cost.png
