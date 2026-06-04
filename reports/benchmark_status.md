# Phase 3 — LLM benchmark status


Total rows aggregated: 4554
Expected: ~4554 (≥ 4189 acceptable)


## Per-combination summary

| Logical name | Prompt | Dataset | Total | OK | Parse fail | API fail | F1 weighted | Accuracy | Median ms | Reasoning tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instant | FS_CoT_RP_NA | employee_reviews | 204 | 204 | 0 | 0 | 0.7306 | 0.7647 | 8345 | 0 |
| llama-3.1-8b-instant | FS_CoT_RP_NA | fakenewsnet | 210 | 209 | 1 | 0 | 0.7083 | 0.7177 | 14434 | 0 |
| llama-3.1-8b-instant | ZS | employee_reviews | 204 | 204 | 0 | 0 | 0.7799 | 0.8088 | 2231 | 0 |
| llama-3.1-8b-instant | ZS | fakenewsnet | 210 | 210 | 0 | 0 | 0.8136 | 0.8143 | 4283 | 0 |
| llama-3.1-8b-instant | ZS_CoT | employee_reviews | 204 | 204 | 0 | 0 | 0.7946 | 0.8186 | 3220 | 0 |
| llama-3.1-8b-instant | ZS_CoT | fakenewsnet | 210 | 210 | 0 | 0 | 0.8233 | 0.8238 | 5275 | 0 |
| llama-3.3-70b-versatile | FS_CoT_RP_NA | employee_reviews | 204 | 204 | 0 | 0 | 0.8622 | 0.8627 | 4308 | 0 |
| llama-3.3-70b-versatile | FS_CoT_RP_NA | fakenewsnet | 210 | 210 | 0 | 0 | 0.8406 | 0.8429 | 6426 | 0 |
| llama-3.3-70b-versatile | ZS | employee_reviews | 204 | 204 | 0 | 0 | 0.8401 | 0.8382 | 2278 | 0 |
| llama-3.3-70b-versatile | ZS | fakenewsnet | 210 | 210 | 0 | 0 | 0.8031 | 0.8095 | 2295 | 0 |
| llama-3.3-70b-versatile | ZS_CoT | employee_reviews | 204 | 204 | 0 | 0 | 0.8501 | 0.8480 | 2275 | 0 |
| llama-3.3-70b-versatile | ZS_CoT | fakenewsnet | 210 | 210 | 0 | 0 | 0.8096 | 0.8143 | 2338 | 0 |
| openai-gpt-oss-120b | FS_CoT_RP_NA | employee_reviews | 204 | 204 | 0 | 0 | 0.8486 | 0.8529 | 3905 | 36859 |
| openai-gpt-oss-120b | FS_CoT_RP_NA | fakenewsnet | 210 | 210 | 0 | 0 | 0.6357 | 0.6524 | 5797 | 46665 |
| openai-gpt-oss-120b | ZS | employee_reviews | 204 | 204 | 0 | 0 | 0.8191 | 0.8186 | 2416 | 16052 |
| openai-gpt-oss-120b | ZS | fakenewsnet | 210 | 210 | 0 | 0 | 0.6566 | 0.6667 | 4548 | 27755 |
| openai-gpt-oss-120b | ZS_CoT | employee_reviews | 204 | 204 | 0 | 0 | 0.8010 | 0.7990 | 3378 | 17161 |
| openai-gpt-oss-120b | ZS_CoT | fakenewsnet | 210 | 210 | 0 | 0 | 0.6609 | 0.6714 | 5249 | 36410 |
| qwen3-32b-reasoning-default | ZS | employee_reviews | 204 | 204 | 0 | 0 | 0.8338 | 0.8325 | 4740 | 47083 |
| qwen3-32b-reasoning-default | ZS | fakenewsnet | 210 | 210 | 0 | 0 | 0.7903 | 0.7905 | 7786 | 69292 |
| qwen3-32b-reasoning-none | ZS | employee_reviews | 204 | 204 | 0 | 0 | 0.8038 | 0.8088 | 2386 | 0 |
| qwen3-32b-reasoning-none | ZS | fakenewsnet | 210 | 210 | 0 | 0 | 0.8333 | 0.8333 | 4433 | 0 |

All combos ≥95% coverage and parse-success.

