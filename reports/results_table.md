# Master results table (condensed)

Best prompt per (model, dataset). Latency = median per call (LLM) or approximate per-record CV mean (baseline).

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
