# Рабочие заметки — Agentic RAG Legal Challenge

## Статус
- [ ] Настроить ANTHROPIC_API_KEY в .env
- [ ] Загрузить корпус: `python scripts/download_corpus.py --phase warmup`
- [ ] Прогнать пайплайн: `python scripts/build_submission.py --phase warmup`
- [ ] Проверить submission.json, отправить
- [ ] Проанализировать результат по типам ответов
- [ ] Установить sentence-transformers если нужен reranker

---

## Формула оценки (ключевое)

```
Total = (0.7 * S_det + 0.3 * S_asst) * G * T * F
```

| Компонент | Что это | Как улучшить |
|-----------|---------|--------------|
| S_det (70%) | Точность детерминированных ответов | LLM extraction > regex |
| S_asst (30%) | LLM-судья по 5 критериям (free_text) | Хороший контекст + краткий ответ |
| G (множитель) | Grounding F-beta β=2.5 (recall >> precision) | Включать все страницы с ответом |
| T (множитель) | Корректность телеметрии | Все поля обязательны |
| F (множитель) | TTFT: <1s=1.05, <2s=1.02, >3s=до 0.85 | claude-haiku быстрый |

**Главный вывод: G — самый опасный множитель.** Плохой retrieval = обнуляет всё остальное.

---

## Сессия 1 — 2026-03-10: No-LLM baseline

### Что сделано
Построен базовый no-LLM пайплайн.

**Файлы:**
- `configs/rag.yaml` — конфиг
- `src/utils/pdf_text.py` — PyMuPDF извлечение текста
- `src/utils/ocr.py` — OCR fallback (pytesseract)
- `src/utils/cache.py` — JSON/pickle кэш
- `src/utils/json_schema.py` — валидация submission.json
- `src/pipeline/ingest.py` — ингестия с OCR fallback
- `src/pipeline/index.py` — BM25Okapi индекс
- `src/pipeline/retrieve.py` — BM25 + соседние страницы ±1
- `src/pipeline/answer.py` — regex/pattern экстракция (без LLM)
- `src/pipeline/telemetry.py` — сборка телеметрии
- `scripts/download_corpus.py` — скачать вопросы + документы
- `scripts/build_submission.py` — оркестратор
- `scripts/submit.py` — отправка
- `src/runners/run_warmup.py` — точка входа

---

## Сессия 2 — 2026-03-10: LLM + Reranker (основной пайплайн)

### Анализ scoring → стратегия

Детально разобрал EVALUATION.md и reference/*.md:
- **G (grounding) — главный риск.** β=2.5 = recall в 2.5× важнее precision. Включать страницы щедрее при ответе.
- **70% S_det** — regex-экстракция даёт ~30-40% точности, LLM даёт ~80-90%. Нужен LLM для всех типов.
- **TTFT < 1s → +5%** — claude-haiku-4-5 обычно 200-500ms на short промпт.
- **Null = правильный ответ** для неответимых вопросов (G=1.0 при обоих пустых).

### Архитектура v2

```
BM25 top-20 + соседи ±1
       ↓
cross-encoder rerank → top-5
       ↓
Claude Haiku (claude-haiku-4-5-20251001) extraction
  - number/bool/date/name/names: structured extraction
  - free_text: grounded synthesis ≤280 chars
  - LLM returns "null" → answer=None, refs=[]
       ↓
_find_source_pages: сопоставление слов ответа с страницами
       ↓
telemetry: ttft==total (no streaming), usage=0
```

### Новые файлы

- `src/pipeline/rerank.py` — cross-encoder реранкинг (sentence-transformers, опционально)
- `src/pipeline/llm.py` — multi-provider LLM клиент (Anthropic > OpenRouter > OpenAI)

### Провайдеры LLM (приоритет)

1. `ANTHROPIC_API_KEY` → `claude-haiku-4-5-20251001` (рекомендуется)
2. `OPENROUTER_API_KEY` → `anthropic/claude-haiku-4-5`
3. `OPENAI_API_KEY` → `gpt-4o-mini`
4. Нет ключей → детерминированный fallback (regex)

### Ключевые решения

| Решение | Обоснование |
|---------|-------------|
| Claude Haiku для всех типов | Быстрый + дешёвый + высокая точность extraction |
| `_find_source_pages` | Сопоставление слов ответа со страницами → точнее grounding |
| `[]` refs при null | Grounding G=1.0 для обоих пустых, правильно по spec |
| Реранкинг опциональный | sentence-transformers ~500MB, но улучшает precision |
| Fallback → regex | Если LLM недоступен, не падаем |

### Потенциальные улучшения (следующий шаг)

- [ ] Запустить на реальных вопросах, посмотреть на какие типы LLM ошибается
- [ ] Проверить что TTFT стабильно < 1s
- [ ] Добавить sentence-transformers reranker (раскомментировать в requirements.txt)
- [ ] Расширить контекст для сложных free_text вопросов
- [ ] Обработка таблиц в PDF — PyMuPDF извлекает таблицы как текст, но плохо

### Известные риски

- LLM может вернуть дату в неверном формате → парсер ловит и возвращает None
- `name`/`names` — LLM иногда возвращает с объяснением → нужна строгая инструкция
- Длинные страницы обрезаются до 2500 символов → ответ может быть в конце страницы
- OCR требует tesseract установленного в системе: `brew install tesseract`

---

## Команды

```bash
# Установка
pip install -r requirements.txt
# Для reranker (опционально, ~500MB):
pip install sentence-transformers

# tesseract для OCR (macOS):
brew install tesseract

# Настройка
export EVAL_API_KEY=...
export EVAL_BASE_URL=https://platform.agentic-challenge.ai/api/v1
export ANTHROPIC_API_KEY=...

# Полный пайплайн
python scripts/download_corpus.py --phase warmup
python scripts/build_submission.py --phase warmup
python scripts/submit.py submission.json code_archive.zip

# Без LLM (только BM25+regex):
python scripts/build_submission.py --phase warmup --no-llm

# Без reranker (только BM25):
python scripts/build_submission.py --phase warmup --no-rerank
```

---

## Сессия 3 — 2026-03-11: Итеративное улучшение на публичном датасете

### Результаты на 100 вопросах (последний прогон)

| Тип | Всего | NULL | NULL% |
|-----|-------|------|-------|
| boolean | 35 | 9 | 26% |
| date | 1 | 1 | 100% |
| free_text | 30 | 0 | 0% |
| name | 14 | 6 | 43% |
| names | 3 | 0 | 0% |
| number | 17 | 12 | 71% |
| **ИТОГО** | **100** | **28** | **28%** |

TTFT avg=1536ms → +2% бонус

### Что было улучшено

1. **Dual-query retrieval для comparison questions**: Boolean вопросы типа "Was Law A enacted in the same year as Law B?" теперь получают страницы из ОБОИХ документов
   - `is_comparison_question()` — детекция сравнительных вопросов
   - `_extract_multi_entity_queries()` — извлечение sub-queries (case numbers / law names)
   - Interleaved page ordering — гарантированное представление обоих entities в контексте
   - `max_per_doc=2` в reranker — diversity cap
   - Увеличили `max_pages=5` для boolean (было 3)
   - **Итог**: boolean 75% NULL → 26% NULL ✓

2. **Law-identity retrieval**: Вопросы о номерах/датах законов теперь получают p.1 документа
   - `_LAW_IDENTITY_RE` детектирует "law number", "title of", "enacted", etc.
   - p.1 всегда содержит "DIFC LAW NO. X OF 20XX"
   - **Итог**: "What is the law number of the Data Protection Law?" → 5 ✓

3. **Article-specific retrieval**: Вопросы с "Article X" ищут эту статью в документе
   - `_ARTICLE_RE` извлекает номер статьи
   - Сканирует все страницы документа на "Article X" и "X." (numbered sections)
   - Добавляет найденные страницы в priority list

4. **`sentence-transformers` установлен** — теперь работает настоящий cross-encoder reranker

### Финальный результат с cross-encoder (top_k=12, pre-warmed)

| Тип | Всего | NULL | NULL% |
|-----|-------|------|-------|
| boolean | 35 | 8 | 23% |
| date | 1 | 1 | 100% |
| free_text | 30 | 0 | 0% |
| name | 14 | 5 | 36% |
| names | 3 | 0 | 0% |
| number | 17 | 10 | 59% |
| **ИТОГО** | **100** | **24** | **24%** |

TTFT avg=2194ms, p50=1657ms → **0% TTFT бонус**

### Влияние cross-encoder reranker (sentence-transformers)

| Метрика | Без reranker | С reranker (top_k=12) |
|---------|-------------|----------------------|
| NULL% | 28% | 24% (-4%) |
| TTFT avg | 1536ms | 2194ms (+658ms) |
| TTFT bonus | +2% | 0% |
| number NULL% | 71% | 59% (-12%) |
| date NULL% | 100% | 100% |
| name NULL% | 43% | 36% (-7%) |

**Вывод**: reranker даёт +4% accuracy но -2% TTFT бонус. Суммарный эффект скорее положительный.

### top_k трейдофф для date вопросов

- "On what date was the Employment Law Amendment Law enacted?" требует найти "Enactment Notice" документ (`bac066...`)
- С top_k=12: документ не попадает в top-12 BM25 → NULL
- С top_k=20: попадает, но LLM видит обрезанный контекст → тоже может быть null
- В последнем успешном прогоне (top_k=20, первый cross-encoder) → ответ **2021-09-14** ✓
- Выбрана компромиссная настройка top_k=15

### Оставшиеся проблемы

| Проблема | Причина |
|----------|---------|
| number 59-71% NULL | Claim value в CA 005/2025 не в orders (нужен весь judgment), article-specific (Employment Law не использует "Article N" формат — использует "N.") |
| date 0-100% NULL | Нестабильно; зависит от top_k |
| boolean 23-26% NULL | Article-specific (Operating Law, GP Law, Personal Property Law); некоторые law comparisons |
| name 36-43% NULL | Article-specific; некоторые comparison cases |
| TTFT 0-2% бонус | CPU reranking ~600-800ms + LLM ~800ms |

### Текущие настройки (rag.yaml)

- top_k_bm25: 15 (компромисс между recall и скоростью)
- top_k_rerank: 5
- reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
- model: claude-haiku-4-5-20251001

### Следующие шаги (когда захочешь продолжить)

1. `python scripts/download_corpus.py --phase warmup` — когда платформа откроет корпус
2. `python scripts/build_submission.py --phase warmup` — собрать submission
3. Commit + push в GitHub
4. Опционально: улучшить article retrieval (Employment Law использует "N." не "Article N")

---

## Сессия 4 — 2026-03-11: Анализ литературы → идеи для улучшений

### Изученные статьи

1. **Habr: Победитель RAG Challenge** (финансовые документы)
   - Per-document BM25/FAISS — один индекс на PDF, роутинг по имени закона/компании
   - Chunk 300 токенов + overlap 50 для retrieval, но LLM получает полную страницу (parent page)
   - LLM reranker: `score = 0.3 * bm25 + 0.7 * llm_score` — точнее cross-encoder
   - Специализированные промпты на каждый answer_type
   - Pydantic валидация + LLM reparser если ответ не прошёл схему
   - Docling (IBM) лучше PyMuPDF для таблиц

2. **A-RAG (2602.03442)** — агент выбирает между keyword/semantic/read инструментами
   - Меньше токенов, лучше recall на сложных вопросах

3. **RAG-Anything (2510.12323)** — multimodal RAG
   - Dual-graph: текст + таблицы/изображения как равноправные узлы
   - Таблицы как первоклассные объекты retrieval — важно для number NULL

4. **RAG Survey (2312.10997)** — таксономия Naive→Advanced→Modular RAG
5. **Original RAG (2005.11401)** — RAG-Token > RAG-Sequence для multi-hop
6. **Agentic RAG Survey (2501.09136)** — валидирует dual-query подход

### Идеи для реализации (приоритет)

| Идея | Источник | Проблема которую решает |
|------|----------|------------------------|
| Per-document BM25 + law name routing | Habr | Precision для single-doc questions |
| LLM reranker (0.3*bm25 + 0.7*llm) | Habr | Замена cross-encoder, лучше precision |
| Chunk 300t + expand to page | Habr | number NULL (ответ в середине страницы) |
| JSON reparser на LLM | Habr | Снизит format errors |
| Таблицы как отдельные retrieval unit | RAG-Anything | number NULL (значения в таблицах) |

---

## Структура репозитория

```
arlc/               # стартовый кит: клиент API, телеметрия
src/
  utils/            # pdf_text, ocr, cache, json_schema
  pipeline/         # ingest, index, retrieve, rerank, answer, llm, telemetry
  runners/          # run_warmup.py
scripts/            # download_corpus, build_submission, submit
configs/rag.yaml    # конфиг
examples/           # примеры из стартового кита (llamaindex, langchain)
reference/          # FAQ организаторов
```
