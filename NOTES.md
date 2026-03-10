# Рабочие заметки — Agentic RAG Legal Challenge

## Статус
- [ ] Загрузить корпус (warmup)
- [ ] Прогнать пайплайн и получить `submission.json`
- [ ] Отправить на оценку
- [ ] Проанализировать результат и улучшить

---

## Сессия 1 — 2026-03-10

### Что сделано

Построен полный no-LLM пайплайн поверх стартового кита.

**Созданные файлы:**
- `configs/rag.yaml` — центральный конфиг (пути, параметры BM25, OCR, fallback-текст)
- `src/utils/pdf_text.py` — постраничное извлечение текста через PyMuPDF
- `src/utils/ocr.py` — опциональный OCR через pytesseract (fallback если текст < 50 символов)
- `src/utils/cache.py` — кэш JSON/pickle на диск
- `src/utils/json_schema.py` — валидация `submission.json` (типы, диапазоны страниц, длина текста)
- `src/pipeline/ingest.py` — ингестия корпуса с OCR-fallback; результат кэшируется
- `src/pipeline/index.py` — BM25Okapi индекс + кэш на диск
- `src/pipeline/retrieve.py` — BM25 поиск + расширение на соседние страницы (±1)
- `src/pipeline/answer.py` — детерминированные экстракторы для всех 6 типов ответов
- `src/pipeline/telemetry.py` — сборка телеметрии (ttft==total_time для no-stream)
- `scripts/download_corpus.py` — скачать вопросы + документы через `arlc.EvaluationClient`
- `scripts/build_submission.py` — оркестратор: ingest → index → retrieve → answer → validate → zip
- `scripts/submit.py` — отправка `submission.json` + `code_archive.zip`
- `src/runners/run_warmup.py` — точка входа для warmup-фазы

**Обновлено:**
- `requirements.txt` — добавлены `pymupdf`, `rank-bm25`, `PyYAML`, `Pillow`, `pytesseract`
- `README.md` — добавлена секция "No-LLM baseline" с командами

### Архитектура

```
download_corpus.py
       ↓
  docs_corpus/<phase>/*.pdf
  data/<phase>/questions.json
       ↓
  ingest.py  →  .cache/<phase>/pages.json
       ↓
  index.py   →  .cache/<phase>/index.pkl  (BM25Okapi)
       ↓
  retrieve.py  →  top-20 страниц + соседи top-5
       ↓
  answer.py  →  детерминированная экстракция по answer_type
       ↓
  telemetry.py  →  timing + retrieval refs
       ↓
  submission.json  +  code_archive.zip
```

### Ключевые решения

| Решение | Обоснование |
|---------|-------------|
| Нет LLM | Требование задачи; детерминированность + скорость |
| BM25 (rank-bm25) | Лёгкое решение без GPU, хорошо работает на юридическом тексте |
| OCR только на пустых страницах | Не тратим время на страницы с текстом |
| `free_text` → фиксированная фраза + пустые refs | `retrieved_chunk_pages=[]` даёт grounding=1.0 на неответимых |
| `ttft_ms == total_time_ms` | Нет стриминга — согласно правилам |
| Только реально использованные страницы в refs | Maximizes grounding precision (F-beta β=2.5) |
| Null + пустые refs при провале экстракции | Правила: unanswerable deterministic → null |

### Потенциальные улучшения

- [ ] Добавить embeddings-реранкинг поверх BM25 (без LLM: sentence-transformers)
- [ ] Улучшить экстракцию `name`/`names` — сейчас работает только по заглавным буквам
- [ ] Добавить нормализацию числовых ответов (миллионы, проценты)
- [ ] Тестирование на реальных warmup-вопросах и анализ точности
- [ ] Оптимизация `top_k_bm25` по результатам оценки

### Известные ограничения

- Экстракция `bool` работает по паттернам; сложные отрицания могут не распознаться
- Экстракция `number` берёт первое число в наиболее релевантном предложении — может быть не то число
- `name`/`names` работают через заглавные буквы — не работает на ALL CAPS текстах

---

## Команды

```bash
# Установка
pip install -r requirements.txt

# Настройка (или через .env)
export EVAL_API_KEY=...
export EVAL_BASE_URL=https://platform.agentic-challenge.ai/api/v1

# Скачать корпус
python scripts/download_corpus.py --phase warmup

# Собрать submission
python scripts/build_submission.py --phase warmup

# Отправить
python scripts/submit.py submission.json code_archive.zip
```

---
