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
