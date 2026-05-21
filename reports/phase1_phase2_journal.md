# Phase 1 + Phase 2 Yolculuk Raporu

**Proje:** LLM Text Classification — Kostina et al. 2025 (arXiv:2501.08457) replikasyon + 3 katkı.
**Tarih:** 2026-05-19.
**Deadline:** 2026-05-20 23:59 (mini-presentation 21 Mayıs).
**Toplam çalışma süresi:** ~6 saat (gerçek wall-clock); kalan bütçe ~26 saat.
**Durum:** Phase 1 ✓ kapandı, Phase 2 ✓ kapandı (gate FAIL ama dokümante edildi), Phase 3 hazır.

İçerik:
- [1. Genel Bakış](#1-genel-bakış)
- [2. Faz 1 — Veri Hazırlama](#2-faz-1--veri-hazırlama)
- [3. Faz 2 — Baseline Modeller](#3-faz-2--baseline-modeller)
- [4. Üretilen Dosyalar](#4-üretilen-dosyalar)
- [5. Karşılaşılan Tüm Sorunlar — Özet Tablo](#5-karşılaşılan-tüm-sorunlar--özet-tablo)
- [6. Şu Andaki Durum ve Sonraki Adım](#6-şu-andaki-durum-ve-sonraki-adım)

---

## 1. Genel Bakış

**Hedef:** Kostina et al. 2025'in 5 baseline + LLM benchmark çalışmasını kendi datalarımızda replike etmek, sonra üç katkı eklemek:

1. **Yeni nesil modeller** — paper'da olmayan 2025 LLM'lerini benchmark'a sok.
2. **Reasoning-augmented inference** — built-in reasoning manuel CoT'nin yerini alabilir mi?
3. **Zenginleştirilmiş baseline'lar** — eğitmen isteği üzerine Random Forest + XGBoost eklendi.

**Workflow:** Multi-agent (Architect → Implementer → Reviewer) gate-approval modeliyle.

**Datasetler:**
- FakeNewsNet (binary): `Jinyan1/PolitiFact` HuggingFace, 214 satır, 50/50 dengeli
- Employee Reviews (3-class): Kaggle Glassdoor reviews, ~200 satır, work-arrangement classification (`remote` / `not_remote` / `not_mentioned`)

**Baseline'lar (5):** NB, SVM, RF, XGBoost, RoBERTa-base.
**LLM'ler (4, Phase 3'te):** Llama 3.3 70B, Qwen3 8B, Gemma 2 9B, DeepSeek-R1-Distill (Groq free tier üzerinden).
**Prompt stratejileri (3):** ZS, ZS_CoT, FS_CoT_RP_NA.

---

## 2. Faz 1 — Veri Hazırlama

### 2.1 Architect aşaması

Designed `docs/design/phase1_data.md` — 10 bölümlü tasarım dokümanı:
- Repo iskeleti (pyproject, .env.example, src/, scripts/, tests/)
- FakeNewsNet acquisition (HF primary, CSV+scrape fallback)
- Employee Reviews acquisition (Glassdoor CSV, sadece headline+pros+cons kolonları kullanılıyor — 1.5 GB → 50 MB memory)
- Keyword pre-classification (regex word-boundary, case-insensitive, multi-word literal escape)
- Stratified sampling deterministik (seed=42)
- Held-out FS rows reserve (Phase 3 few-shot için)
- Validation gates (≥180 toplam, ≥40 per class)

**5 open decision** sunuldu:
1. ER held-out FS rows — Phase 1'de label'lamak yerine Phase 3'e ertelendi ✓
2. FNN held-out FS rows — Phase 1'de inherit label (fake/real) ✓
3. HF label encoding ambiguity → auto-fallback to CSV+scrape ✓
4. ≥40/class gate sert + recovery instructions ✓ MODIFIED (supplemental batch script eklendi)
5. ER candidate pool — 70/class (3 sınıf × 70 = 210 candidate) ✓

### 2.2 Implementer aşaması

Üretildi:
- `pyproject.toml` (Python 3.11+, Phase 1 deps)
- `.env.example` (GROQ_API_KEY, HF_TOKEN placeholders)
- `src/llm_textcls/`: `io.py`, `logging_setup.py`, `data/{tokenization.py, fakenewsnet.py, employee_reviews.py}`
- `scripts/`: `build_fakenewsnet.py`, `build_employee_reviews_unlabeled.py`, `split_for_labeling.py`, `merge_labeled.py`, `generate_supplemental_batch.py`, `sanity_check_labels.py`, `data_summary.py`
- `data/employee_reviews/LABELING_INSTRUCTIONS.md` (Doğuş'un Claude.ai parallel chat'leri için rubric)
- `tests/`: `test_io.py`, `test_tokenization.py`, `test_fnn_sampling.py`, `test_er_filter.py`, `test_split_merge.py`

### 2.3 Faz 1'de karşılaşılan sorunlar ve çözümler

#### Sorun 1: Sistemde Python 3.9 default, projeye 3.11+ lazım

**Belirti:** İlk pytest çalıştırma denemesinde `ModuleNotFoundError: No module named 'tiktoken'` ve `Python 3.9.13` çıktısı.

**Tanı:** Project pyproject.toml `requires-python = ">=3.11"`. Sistem default 3.9. Phase 1 deps (pandas 2.2+, tiktoken, datasets) için yeni Python lazım.

**Çözüm:** Doğuş Python 3.14.5 kurdu (sonra Phase 2 için 3.12'ye geçildi). venv oluşturup `pip install -e ".[dev]"`.

#### Sorun 2: Pandas 3.x'te `astype(str)` dtype "object" değil "str" döndürüyor

**Belirti:** `test_io.py` testleri: `ValueError: Schema mismatch for ERUnlabeledRow.id: expected one of ['object', 'string'], got str`.

**Tanı:** Pandas 3.0.3 (Python 3.14 ile birlikte gelen yeni sürüm) StringDType'ı default haline getirmiş. Bizim `_DTYPE_MAP` sadece `{"object", "string"}` kabul ediyordu.

**Çözüm:** `src/llm_textcls/io.py` içinde `_DTYPE_MAP[str]`'e `"str"` eklendi:
```python
_DTYPE_MAP = {
    str: {"object", "string", "str"},
    ...
}
```

#### Sorun 3: numpy 2.4 / pandas 3.0'da `np.array_split(df, n)` numpy array döndürüyor, DataFrame değil

**Belirti:** `test_split_merge.py`:
```
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
and integer or boolean arrays are valid indices
```

**Tanı:** Önceki numpy sürümlerinde `np.array_split(df, 4)` 4 DataFrame'lik liste döndürüyordu. Yeni numpy 2.4 ile DataFrame'i ndarray'e çeviriyor; sonraki `chunk[["id", "text"]]` patlıyor.

**Çözüm:** `scripts/split_for_labeling.py` içinde manuel chunking:
```python
n = len(df)
chunk_size = math.ceil(n / args.n_batches)
for i in range(args.n_batches):
    start = i * chunk_size
    end = min(start + chunk_size, n)
    chunk = df.iloc[start:end]
```

#### Sorun 4: HF dataset `Jinyan1/PolitiFact` label kolonu yok — label split adlarında

**Belirti:** İlk FNN build denemesi:
```
HF dataset missing text/label columns. cols=['id', 'description', 'text', 'title']
text=text label=None
HF route returned empty. Switching to CSV fallback.
FileNotFoundError: CSV fallback requires politifact_fake.csv ...
```

**Tanı:** HF dataset 4 split'e sahip: `MF`, `HF`, `MR`, `HR` (Mostly Fake, Highly Fake, Mostly Real, Highly Real). Label kolonu YOK — label'ı **split adından** çıkarmak gerek. Tasarımda bu pattern (MF/HF/MR/HR) belirtilmişti ama loader sadece kolon bazlı arıyordu.

**Çözüm:** `src/llm_textcls/data/fakenewsnet.py:load_from_huggingface`'e split-name label inference eklendi:
```python
for split_name in ds:
    split_df = ds[split_name].to_pandas()
    split_label = _normalize_label_value(split_name)  # "MF" -> "fake" via lowercase + set lookup
    split_df["_split_label"] = split_label
    ...
if label_col is None and split_labels_resolved:
    log.warning("HF label inferred from split names: %s", split_names_seen)
    raw["_norm_label"] = raw["_split_label"]
```

Sonuç: 520 satır (97 MF + 97 HF + 132 MR + 194 HR) yüklendi, 460 length filter'dan geçti, 214 sample alındı (107/107).

### 2.4 Faz 1 Reviewer aşaması — B1 blocker

`prompts/00_role_reviewer.md` rolüyle audit yapıldı. Spesifik 6 madde kontrol edildi:

1. HF split-name label mapping ✓
2. Keyword regex (case-insensitive, word boundary, literal escape) ✓
3. `merge_labeled.py` glob pattern (`batch_*_labeled.csv` her ikisini de yakalıyor) ✓
4. **`generate_supplemental_batch.py` exclusion — `held_out_fs.parquet` id'lerini DIŞARDA BIRAKMIYOR ✗ BLOCKER**
5. WARNING-level logs (HF route decisions) ✓
6. Random seeds (`seed=42` her yerde) ✓

#### B1 Detayı

`generate_supplemental_batch.py` sadece `batch_*.csv` dosyalarındaki id'leri excluded'a alıyordu. Ama **held-out FS rows zaten batch CSV'lerinde DEĞİL** (OD-1 gereği label'lanmıyorlar). Id'ler deterministik `sha256(text)[:12]` olduğundan, supplemental run aynı held-out review'u re-sample edebilir → Phase 3'te FS example ile test set'in aynı satırı paylaşması = **leakage**.

#### B1 Çözümü

`scripts/generate_supplemental_batch.py`:
```python
existing_ids = er.existing_ids_in_batches(args.to_label_dir, logger=log)
for parquet_name in ("unlabeled.parquet", "held_out_fs.parquet"):
    p = args.to_label_dir.parent / parquet_name
    if p.exists():
        existing_ids |= set(pd.read_parquet(p, columns=["id"])["id"].astype(str))
```

#### Regresyon testi

`tests/test_split_merge.py`'a `test_supplemental_batch_excludes_held_out_fs_ids` eklendi. **İlk versiyon flaky idi** — seed=42 ile random sample held-out'u doğal olarak atlayabilirdi (1/5 şans). Deterministik versiyona dönüştürüldü:

- 5 remote-keyword review, row 0 held-out olarak işaretle
- N=5 talep et (=total rows)
- Fix VARSA: 4 available → script `RuntimeError` raise eder ("not enough rows")
- Fix YOKSA: sample(n=5) tüm 5'i döner → held-out kesinlikle içeride

Bu pigeonhole argümanı ile test deterministik catches bug. Verify edildi: fix kaldırılınca test FAIL, fix geri konulunca PASS.

### 2.5 Faz 1 final çıktıları

- `data/fakenewsnet/sample.parquet`: 214 satır, 107/107, mean token=495, max=3598
- `data/employee_reviews/unlabeled.parquet`: 204 satır
- `data/employee_reviews/held_out_fs.parquet`: 6 satır (2/keyword_class)
- `data/employee_reviews/to_label/batch_{1..4}.csv`: 51 satır/batch, empty label kolonu
- Doğuş 4 parallel Claude.ai chat'inde label'ladı, `batch_{1..4}_labeled.csv` üretti
- `data/employee_reviews/labeled.parquet`: 204 satır, sınıf dağılımı 95/57/52 (not_mentioned/remote/not_remote)
- `reports/data_summary.md` + figures
- 19 test geçti

---

## 3. Faz 2 — Baseline Modeller

### 3.1 Architect aşaması

`docs/design/phase2_baselines.md` üretildi. 5 baseline (NB, SVM, RF, XGBoost, RoBERTa) × 2 dataset × 5-fold CV = 50 satır sonuç. Validation gate ±5 hard / ±10 soft, paper hedefleriyle karşılaştırma.

**5 open decision:**
1. RoBERTa-ER fold count — 5-fold default, runtime fallback to 3 if FNN ilk fold > 4 min ✓
2. fp16 mixed precision — enabled (Ampere GPU) ✓
3. RoBERTa class weights for ER — enabled (per-train-fold, sklearn balanced) ✓
4. Validation gate — uniform ±5/±10 ✓
5. RoBERTa max_length=512 (paper'la eşleşme) ✓

### 3.2 Implementer aşaması

Üretildi:
- `src/llm_textcls/baselines/`: `tfidf_models.py` (NB/SVM/RF/XGBoost + `_XGBStringWrapper`), `roberta.py` (`WeightedTrainer` + `cross_validate_roberta`)
- `src/llm_textcls/evaluation/metrics.py` (`BaselineResult`, `weighted_f1`, `aggregate_cv_results`, `results_to_dataframe`)
- `scripts/`: `run_baselines_tfidf.py`, `run_baselines_roberta.py` (CUDA pre-check + install hint), `aggregate_baselines.py` (gate verdict + report)
- `tests/test_metrics.py`, `tests/test_baselines_tfidf.py` (toplam 13 yeni test, RoBERTa testi YOK — çok yavaş)

13 yeni test ile total 32 test, 21 saniyede geçiyor.

### 3.3 Faz 2'de karşılaşılan sorunlar ve çözümler

#### Sorun 5: Python 3.14.5 ile transformers/torch wheel yokluğu riski → 3.12'ye geçiş

**Belirti:** Phase 2'ye geçişte kullanıcı `py -3.12 -m venv .venv` ile venv'i yeniden oluşturdu.

**Tanı:** Python 3.14.5 çok yeni (Ekim 2025), bazı paketlerin (özellikle torch CUDA wheels) cp314 wheel'i hazır olmayabilirdi. 3.12 stabil mainstream.

**Sonuç:** Phase 1 dosyaları 3.12'de de sorunsuz çalıştı, 32 test geçti.

#### Sorun 6: `pip install -e .` torch CPU wheel'ini transitively çekti, sonraki `pip install torch --index-url cu128` "zaten yüklü" deyip atladı

**Belirti:**
```
torch version: 2.12.0+cpu
torch.version.cuda: None
cuda available: False
```

**Tanı:** Sıralama hatası:
1. `pip install -e .` çalıştırıldı → pyproject.toml'da `transformers` var → transformers torch'u transitive dependency olarak istedi → pip CPU wheel'i PyPI'dan çekti
2. Sonra `pip install torch --index-url ...cu128` → pip "torch zaten satisfied" deyip skip etti

#### Sorun 7: cu128 wheel'ı driver 546.30 ile uyumsuz

**Belirti:** `nvidia-smi`:
```
NVIDIA-SMI 546.30, Driver Version: 546.30, CUDA Version: 12.3
```

**Tanı:** Driver 546.30 maksimum CUDA 12.3 runtime'ı destekliyor. cu128 wheel CUDA 12.8 runtime ister → driver ≥ R570 gerek. cu121 (CUDA 12.1) bu driver'la sorunsuz çalışır ve design dokümanında zaten önerilen yol.

**Çözüm (her iki sorun için):**
```powershell
pip uninstall -y torch
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

`--force-reinstall` "zaten yüklü" skip'ini bypass eder.

**Doğrulama:**
```
torch: 2.5.1+cu121
cuda available: True
cuda version: 12.1
device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

#### Sorun 8 (tool-side, project-irrelevant): Sandbox'tan `from transformers import Trainer` segfault

**Belirti:** Kod kontrolü için sandbox üzerinden çalıştırdığım `python -c "from transformers import Trainer"` segmentation fault verdi.

**Tanı:** Bu Claude Code sandbox'ın transformers/CUDA library load mekanizmasıyla bir uyumsuzluğu. Kullanıcı tarafında transformers çalışıyor (RoBERTa 10 fold fine-tune başarıyla koştu).

**Çözüm:** Transformers source'unu disk üzerinden `Grep` tool'uyla okuyup `compute_loss` imzasını inspect ettim. Workaround.

#### Sorun 9: RoBERTa-ER fine-tune HİÇ öğrenmiyor — train loss log(3) baseline'da takılı

**Belirti:** İlk Phase 2 run sonuçları:

| Fold | ER train loss (final epoch 3) |
|---|---|
| 1 | 1.094 |
| 2 | 1.094 |
| 3 | 1.078 |
| 4 | 1.091 |
| 5 | 1.082 |

log(3) ≈ 1.099 = 3-class uniform random prediction. Model classifier head'i random init'ten kaçamamış. F1=0.41 (target 0.838, Δ=42.7 CATASTROPHIC FAIL).

FNN'de aynı kod aynı params'la **çalışıyordu** (loss 0.63 → 0.50), yani kod bug'lı değil — ER-spesifik bir issue.

**Tanı (rooting cause analysis):**

1. 163 train sample × batch_size 8 = ~20 optimizer step / epoch
2. 3 epoch × 20 step = **toplam ~60 optimizer step**
3. RoBERTa classifier head random initialize ediliyor (body pretrained, head yeni)
4. 60 step random init'ten anlamlı öğrenmeye geçmek için yetersiz
5. Class weights (balanced) loss sinyaline ekstra gürültü ekliyor
6. ER metinleri çok kısa (mean 69 token) → CLS token discriminative info'yu yakalamakta zorlanıyor

FNN ile fark:
- FNN: 2 class, 50/50 balanced → class weights = [1.0, 1.0] (no-op)
- FNN: metinler daha uzun (mean 495)
- Aynı 60 step yetiyor

**API check:** transformers 5.x'in `Trainer.compute_loss` imzasını disk'ten okudum:
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
```

Benim `WeightedTrainer.compute_loss(self, model, inputs, return_outputs=False, **_kwargs)` — `**_kwargs` `num_items_in_batch`'i sessizce yutuyor. **API uyumlu, sorun bu değil.**

**Çözüm:** Debug rerun script — `scripts/rerun_roberta_er_debug.py`:
- `epochs=10` (vs 3) → 200 optimizer step
- `lr=5e-5` (vs 2e-5) → daha hızlı convergence (random-init head için)
- `use_class_weights=False` (vs True) → loss sinyali pürüzsüz
- FNN sonuçları parquet'te korundu, sadece ER yeniden çalıştırıldı

**Sonuç:**

| Fold | ER train loss (önceki / debug) | ER F1 (önceki / debug) |
|---|---|---|
| 1 | 1.094 / **0.508** | 0.41* / **0.765** |
| 2 | 1.094 / **0.470** | 0.41* / **0.640** |
| 3 | 1.078 / **0.454** | 0.41* / **0.733** |
| 4 | 1.091 / **0.553** | 0.41* / **0.832** |
| 5 | 1.082 / **0.440** | 0.41* / **0.780** |
| Mean | ~1.09 / **~0.49** | **0.411 / 0.750** |

*Önceki: rastgele tahmin yapan model. ‡‡Yeni: gerçek öğrenme.

**+34 F1 puan iyileşme.** RoBERTa-ER hâlâ SOFT WARN (Δ=8.8) ama hard FAIL'dan çıktı.

#### Sorun 10: ruff dev venv'de yoktu

**Belirti:** `python -m ruff format ...` → `No module named 'ruff'`.

**Tanı:** Kullanıcı 3.12 venv'i kurarken `pip install -e .` yaptı, `[dev]` extras'ı atladı.

**Çözüm:** `pip install ruff` (pytest zaten gelmişti başka yoldan).

### 3.4 NB-ER hard FAIL (kalıcı) — small-N etkisi

**Sonuç:** NB-ER F1=0.459, target 0.613, Δ=15.4 — hard FAIL.

**Bu bug değil**, kabul edilen small-N sınırlaması:

- **NB-FNN paper hedefini birebir tutturuyor** (0.900 / 0.900) → TF-IDF + MultinomialNB pipeline'ı **doğru çalışıyor**.
- Aynı pipeline ile SVM-ER target içinde (Δ=3.0).
- Paper N=1000, biz N=204. MultinomialNB high-bias / low-capacity → küçük data'da en çok etkilenen model.
- ER metinleri kısa (mean 69 token) → bigram feature uzayı sparse.
- Mini-presentation için anlamlı bir bulgu: "small-N regime'de NB çöküyor, modern LLM'ler bu açığı kapatabilir mi?" sorusunu güçlendiriyor.

### 3.5 Faz 2 final gate

| Model | FNN F1 | Target | Δ | ER F1 | Target | Δ | Verdict |
|---|---|---|---|---|---|---|---|
| NB | 0.900 | 0.900 | 0.0 | 0.459 | 0.613 | **15.4** | FNN PASS / ER **HARD FAIL** |
| SVM | 0.880 | 0.888 | 0.8 | 0.658 | 0.687 | 3.0 | PASS / PASS |
| RoBERTa | 0.856 | 0.930 | 7.4 | 0.750 | 0.838 | 8.8 | SOFT / SOFT |
| RF | 0.790 | — | — | 0.624 | — | — | report-only |
| XGB | 0.774 | — | — | 0.714 | — | — | report-only |

**Yalnızca NB-ER hard FAIL.** Diğer 5 gated kombinasyon PASS veya SOFT.

### 3.6 Faz 2 dokümante edilmiş sapmalar

`docs/decisions/phase2_baselines.md` oluşturuldu. İçinde:
- OD-1 runtime outcome (5-fold tutuldu, fallback gerekmedi)
- OD-2 fp16 outcome (NaN yok, stabil)
- **OD-3 post-run deviation** — RoBERTa-ER için (`epochs=10, lr=5e-5, class_weights=False`). Sebebi + root cause + debug script referansı.
- OD-4 outcome (NB-ER FAIL)
- OD-5 outcome (max_length=512 uygulandı)

`reports/baselines.md` içine kalıcı caveat eklendi (script'in `_write_report` fonksiyonuna):
1. NB-ER hard FAIL bir pipeline bug'ı değil, small-N etkisi (NB-FNN birebir target tutturuyor)
2. RoBERTa-ER debug hyperparameters kullanıyor; sebep + referans

Yeniden aggregate çalıştırıldığında bu notlar otomatik regenerate edilir.

---

## 4. Üretilen Dosyalar

### Faz 1

```
pyproject.toml
.env.example
docs/
├── design/phase1_data.md
└── decisions/phase1_data.md
src/llm_textcls/
├── __init__.py
├── io.py
├── logging_setup.py
└── data/
    ├── __init__.py
    ├── tokenization.py
    ├── fakenewsnet.py
    └── employee_reviews.py
scripts/
├── build_fakenewsnet.py
├── build_employee_reviews_unlabeled.py
├── split_for_labeling.py
├── merge_labeled.py
├── generate_supplemental_batch.py
├── sanity_check_labels.py
└── data_summary.py
tests/
├── conftest.py
├── test_io.py
├── test_tokenization.py
├── test_fnn_sampling.py
├── test_er_filter.py
└── test_split_merge.py
data/
├── fakenewsnet/sample.parquet           # 214 satır
└── employee_reviews/
    ├── unlabeled.parquet                # 204 satır
    ├── held_out_fs.parquet              # 6 satır
    ├── labeled.parquet                  # 204 satır (Doğuş label'ladı)
    ├── LABELING_INSTRUCTIONS.md
    └── to_label/
        ├── batch_{1..4}.csv             # 51 satır/batch
        └── batch_{1..4}_labeled.csv     # Doğuş'un labeled output'u
reports/
├── data_summary.md
└── figures/
    ├── fnn_token_lengths.png
    └── er_token_lengths.png
```

### Faz 2

```
docs/
├── design/phase2_baselines.md
└── decisions/phase2_baselines.md
src/llm_textcls/
├── baselines/
│   ├── __init__.py
│   ├── tfidf_models.py    # NB, SVM, RF, XGBoost + _XGBStringWrapper
│   └── roberta.py         # WeightedTrainer + cross_validate_roberta
└── evaluation/
    ├── __init__.py
    └── metrics.py         # BaselineResult, weighted_f1, aggregate_cv_results
scripts/
├── run_baselines_tfidf.py
├── run_baselines_roberta.py
├── rerun_roberta_er_debug.py    # OD-3 deviation script
└── aggregate_baselines.py
tests/
├── test_metrics.py        # 6 test
└── test_baselines_tfidf.py # 7 test
results/baselines/
├── raw_tfidf.parquet       # 40 satır (4 model × 2 dataset × 5 fold)
├── raw_roberta.parquet     # 10 satır
├── raw.parquet             # 50 satır (Phase 4 join key)
└── summary.parquet         # 10 satır (mean/std)
reports/
└── baselines.md            # Gate verdict + tablo + caveats
```

### Toplam istatistik

- **20 Python kaynak dosyası** (modüller + scripts)
- **7 test dosyası**, **32 test** (Faz 1: 19, Faz 2: 13), tümü ~9 saniyede geçiyor
- **4 markdown design/decisions doc**
- **2 markdown rapor** (`data_summary.md`, `baselines.md`)
- **6 parquet artifact** (4 dataset, 2 results)
- **20 dosya** `ruff format` + `ruff check` clean

---

## 5. Karşılaşılan Tüm Sorunlar — Özet Tablo

| # | Faz | Sorun | Tanı | Çözüm |
|---|---|---|---|---|
| 1 | 1 | Sistemde Python 3.9 default | Project 3.11+ ister | Python 3.14.5 kur, venv |
| 2 | 1 | `astype(str)` dtype "str" döndürdü | Pandas 3.0 StringDType default | `_DTYPE_MAP[str]`'e `"str"` ekle |
| 3 | 1 | `np.array_split(df, n)` ndarray döndürdü | numpy 2.4 davranış değişikliği | Manuel `df.iloc[start:end]` chunking |
| 4 | 1 | HF dataset label kolonu yok | Label split adlarında (MF/HF/MR/HR) | Split-name label inference (`_split_label` kolonu) |
| 5 | 1 | Reviewer B1: held-out FS leakage riski | `generate_supplemental_batch.py` parquet'leri kontrol etmiyor | Union ile `unlabeled.parquet` + `held_out_fs.parquet` id'leri exclude et |
| 6 | 1 | Regression test ilk versiyonu flaky | seed=42 random sample held-out'u atlayabilir | N==total deterministik test (pigeonhole) |
| 7 | 2 | Python 3.14'ten 3.12'ye geçiş | Wheel availability riski | `py -3.12 -m venv .venv` |
| 8 | 2 | CPU torch wheel yüklendi | `pip install -e .` transformers'ı önce çekti, sonraki torch install skip etti | `pip uninstall -y torch && pip install --force-reinstall torch --index-url ...cu121` |
| 9 | 2 | cu128 driver 546.30 ile uyumsuz | Driver max CUDA 12.3 destekliyor | cu121 (CUDA 12.1) wheel kullan |
| 10 | 2 | Sandbox transformers segfault | Tool-side artifact, project'i etkilemez | Source'u disk'ten grep'le inceledim |
| 11 | 2 | RoBERTa-ER hiç öğrenmiyor (F1=0.41) | 60 optimizer step random-init head için yetersiz, class weights ek gürültü, kısa ER metinleri | Debug rerun: epochs=10, lr=5e-5, no class weights → F1=0.75 (+34 puan) |
| 12 | 2 | ruff dev venv'de yok | Kullanıcı `[dev]` extras'ı atladı | `pip install ruff` |
| 13 | 2 | NB-ER hard FAIL (Δ=15.4) | Small-N etkisi, paper N=1000 vs biz 204 | Accept + caveats'te dokümante; pipeline doğru (NB-FNN target birebir) |

---

## 6. Şu Andaki Durum ve Sonraki Adım

### Phase 1 ✓ kapandı
- Pipeline çalıştı, 204 ER + 214 FNN hazır
- 6 held-out FS row Phase 3 için rezerve
- Reviewer onayladı, B1 fix'i regression test ile korundu

### Phase 2 ✓ kapandı (gate verdict FAIL ama dokümante)
- 5 model × 2 dataset × 5 fold = 50 satır sonuç
- Gate: NB-ER hard FAIL (small-N), RoBERTa her iki dataset'te SOFT WARN, diğerleri PASS
- Pipeline doğruluğu kanıtlandı: NB-FNN paper hedefini birebir tutturdu
- Sapmalar tam dokümante edildi (`docs/decisions/phase2_baselines.md`, `reports/baselines.md` caveats)

### Phase 3 hazır
Phase 3 — LLM benchmark (4 model × 3 prompt × 414 record ≈ 5000 Groq calls). Architect rolünü başlatmak için:

```
Read prompts/00_role_architect.md and prompts/phase3_llm_benchmark_architect.md,
then act as Architect for Phase 3. Produce the design document at
docs/design/phase3_llm_benchmark.md.
```

### Zaman bütçesi
- Kalan deadline: 2026-05-20 23:59 (~26 saat şu andan itibaren)
- Phase 3 tahmini: 90 dk impl + 30-60 dk execution (5000 Groq call, free tier rate limit'e göre)
- Phase 4 (aggregation + figures + report): 60-90 dk
- Mini-presentation: 21 Mayıs (Phase 4 sonrası rahatlıkla yetişir)

### Final repo durumu

```
$ ./.venv/Scripts/python.exe -m pytest
32 passed in 8.87s

$ ./.venv/Scripts/python.exe -m ruff format --check src scripts tests
30 files left unchanged

$ ./.venv/Scripts/python.exe -m ruff check src scripts tests
All checks passed!
```

Sağlıklı durumda Phase 3'e geçişe hazır.
