# Text-to-SQL Data Analyst

Upload a data file, ask questions about it in plain English and get back a
table, a chart and a short written answer. A local language model (Ollama)
writes the SQL, SQLite runs it, and the app corrects its own failed queries.
Nothing leaves your machine.

## Problem Statement

Business users often have their data in spreadsheets and exports (CSV, Excel,
JSON), but answering even simple questions such as "Which region sold the most
last quarter?" needs either SQL skills or spreadsheet work. These files are
usually messy too: title rows above the header, numbers stored as text,
"N/A" placeholders, mixed date formats and inconsistent spellings. Cloud AI
tools can help, but sending company data to an outside service is often not
allowed.

This project is a GenAI data analyst that:

1. **Cleans the uploaded file automatically** and reports what it changed and
   what looks suspicious.
2. **Turns plain-English questions into SQL** with a local LLM, runs the SQL
   safely against SQLite and **retries with the error message** when a query
   fails.
3. **Explains the result** with a chart chosen from the data's shape and a
   short summary whose numbers are checked against the result.
4. **Remembers the conversation**, so follow-ups such as "only for Laptop"
   work.
5. **Protects the data** with guardrails against prompt injection, write
   queries and access to anything other than the uploaded tables.

## Features

| Area | What it does |
|---|---|
| File types | CSV, TSV, Excel (.xlsx, one table per sheet), JSON, Parquet; up to 50 MB |
| Auto cleaning | Header detection, snake_case column names, null placeholders, numbers/dates/booleans stored as text, thousands separators, dates typed in many formats ("23rd December of 1966", "22 Feb 04"; each guess is listed), merged spelling variants, safe duplicate removal. Cells holding several values ("Crime, Drama") get a second table with one row per value, joined on `row_id`. The original values are kept in a `<table>_raw` table |
| Cleaning review | Right after upload, before the first question: the judgment calls the cleaning left open (missing values, look-alike values, numbers stored as text, guessed dates, list columns, extreme values), each with options and a suggested pick. **Apply** re-cleans the original file with the choices; **Skip** keeps the automatic cleaning. Decided from values and types, never column names; no model call. Can be changed later in the Data tab, or from the chat: "fill age with the median" or "drop rows with missing email" shows a confirm button, then re-cleans the same way |
| Text-to-SQL | Ollama `qwen2.5` (7B on GPU, 3B on CPU, chosen at startup), JSON output (plan, can_answer, sql), few-shot prompt with the dataset's schema |
| Self-correction | Errors and failed checks go back to the model, up to 3 retries; an empty result gets a second look; the loop stops when the model repeats any query it already tried; refusals are final |
| Structure questions | "What columns are there?", "describe the data", "list the fields" are answered from the profile without SQL or a model call: one row per column (type, example, missing) plus 3–5 suggested questions built from the column types. If the model ever writes `pragma_table_info` / `sqlite_master` SQL (always blocked), the same answer is returned instead of retrying |
| Guardrails | Question checks (length, injection phrases, raw SQL), static SQL check (single SELECT/WITH, no write keywords, allowed tables only), read-only connection with a SQLite authorizer, 10 s query timeout, 1,000-row cap, a 240 s time budget per question (no new model call after it) |
| Charts | Rule-based from the result's shape: stat tile, line, small multiples, bar, grouped bar, scatter, histogram. Ask for a type ("as a pie chart", donut, column, area, stacked, line, bar, scatter, histogram) and it is drawn when the result fits it, otherwise a note says why. Lists of rows stay a table (Plotly, light and dark palettes) |
| Summary | 1–3 sentences (a longer one is sent back to be shortened); numbers are given to the model pre-formatted and every number must appear in the result, otherwise a fixed template is used. Caveats come from the cleaning flags. Summary, caveats and chart are stored with each turn |
| Chat memory | Chats per dataset; follow-ups rewritten into standalone questions. Messages that are not data questions are sorted out first: "what" or "that's wrong" get a reply asking what to change (status `clarify`, no SQL), "make it a pie chart" redraws the previous answer without new SQL, "top 5 rows" shows the stored rows without a model call, and "delete the North rows" is refused before any SQL is written |
| Evaluation | Held-out questions, follow-ups, hostile prompts and messy files |
| API | FastAPI with Swagger UI and a Postman collection |
| UI | Streamlit app that talks only to the API |

## Tech Stack

Python · Pandas · SQLite · Ollama (qwen2.5-instruct) · Plotly · FastAPI ·
Uvicorn · Streamlit · pytest

## Folder Structure

```
LocalAnalyst/
├── config/
│   └── config.py              paths, model profiles, limits, cleaning thresholds
├── data/
│   ├── input_data/            sample messy files, dev questions, eval/ (held-out)
│   ├── logs/                  app.log
│   └── output_data/evaluation/latest_run/   evaluation reports
├── datasets/                  one workspace per upload (created at runtime)
├── notebook/                  01–08 prototype notebooks, one per stage
├── postman/                   Postman collection for every route
├── src/
│   ├── api/                   one file per route + common.py
│   ├── evaluation/            run_evaluation.py, result_matching.py
│   ├── helpers/               one file per pipeline stage + pipeline.py
│   ├── main/main.py           FastAPI application
│   └── ui/streamlit_app.py    Streamlit frontend
├── tests/                     pytest suite (no model needed)
├── .env.example
├── project_architecture.txt   detailed design notes
├── pytest.ini
├── requirements.txt
├── scripts/ensure_ollama.bat   starts Ollama when needed (used by the .bat files)
└── setup.bat, run_all.bat, run_api.bat, run_ui.bat   Windows: one-time setup, then start
```

The code uses plain functions and dictionaries only (no classes). Each stage
was prototyped in a notebook and then converted to a module:

| Notebook | Module(s) |
|---|---|
| 01_loading_cleaning | `file_loader.py`, `data_cleaner.py` |
| 02_sqlite_profiling | `sqlite_store.py`, `data_profiler.py` |
| 03_model_setup | `model_client.py`, `model_selector.py` |
| 04_text_to_sql | `sql_generator.py` |
| 05_self_correction_guardrails | `input_guardrails.py`, `sql_guardrails.py`, `sql_executor.py`, `pipeline.answer_question` |
| 06_charts_summary | `chart_builder.py`, `answer_summary.py` |
| 07_memory | `chat_memory.py`, `pipeline.ask_in_chat` |
| 08_evaluation | `src/evaluation/run_evaluation.py` |

## How It Works

```
Upload ─► read file ─► auto clean ─► SQLite (clean + raw tables) ─► profile + schema text

Question ─► question guardrail ─► rewrite follow-up (chat memory)
         ─► LLM writes SQL ─► SQL guardrail ─► read-only execution
              ▲                                      │
              └──── error / empty result ◄───────────┘   (self-correction)
         ─► chart (rules) ─► summary (number-checked) + caveats ─► stored in chat
```

## Setup

### On a new Windows computer (recommended)

1. **Install Python 3.11 or newer** if it is not there yet:
   [python.org/downloads](https://www.python.org/downloads/) (tick *Add python.exe to
   PATH*), or in a terminal:

   ```bash
   winget install Python.Python.3.12
   ```

2. **Download the project** from GitHub (*Code → Download ZIP*, then unzip), or:

   ```bash
   git clone <repository-url>
   ```

3. **Double-click `setup.bat`.** It does everything else, and is safe to run again:

   | Step | What it does |
   |---|---|
   | Python | Finds Python 3.11+ (`py` or `python`) and says how to install it if missing |
   | Packages | Creates `.venv` and installs `requirements.txt` (a few minutes the first time; retries without pip's cache if a download is corrupt) |
   | Settings | Creates `.env` from `.env.example` |
   | Ollama | Installs it with winget if you agree, and starts it |
   | Model | Downloads the model that fits: an NVIDIA GPU with 6 GB or more gets `qwen2.5:7b-instruct` (about 4.7 GB), anything else `qwen2.5:3b-instruct` (about 1.9 GB); `MODEL_PROFILE` in `.env` is set to match |
   | Self-test | Runs the test suite (about 15 seconds, no model needed) |

4. **Double-click `run_all.bat`.** It starts Ollama if needed, then the API and the
   app, and opens the app in the browser. `run_all.bat` runs `setup.bat` first if it
   has never been run.

### Manual setup (any system)

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com).

1. Pull a model (the 3B is enough without a GPU) and set `MODEL_PROFILE` to match
   (`gpu` for the 7B, `cpu` for the 3B; `auto` needs both):

   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

2. Create the virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```
   ```bash
   pip install -r requirements.txt
   ```

   On macOS/Linux, activate with `source .venv/bin/activate`.

3. Copy `.env.example` to `.env` and adjust if needed:

   | Variable | Default | Meaning |
   |---|---|---|
   | `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
   | `MODEL_PROFILE` | `auto` | `auto` (7B if it fits fully on the GPU, else 3B; needs both models), `gpu` (7B) or `cpu` (3B) |
   | `DATASETS_DIR` | `datasets/` | Where uploaded datasets are stored |

## Running

### Windows shortcuts

Double-click these in the project folder:

| File | What it does |
|---|---|
| `setup.bat` | One-time setup: creates `.venv`, installs `requirements.txt`, copies `.env.example` to `.env`, pulls the Ollama models |
| `run_all.bat` | Starts the API, waits until it is ready, then starts the UI and opens Swagger UI |
| `run_api.bat` | Starts only the API (http://127.0.0.1:8765) |
| `run_ui.bat` | Starts only the UI (http://localhost:8501); the API must be running |

Ollama must be running first. Close the API and UI windows to stop them.

### Manually

Start Ollama first, then:

**1. API**

```bash
python -m uvicorn src.main.main:app --reload
```

On startup the API checks Ollama and picks the model; it stops with a clear
error if Ollama is down or the model is missing.

- Swagger UI: http://127.0.0.1:8765/docs
- Health check: http://127.0.0.1:8765/

**2. UI** (in a second terminal, with the API running)

```bash
python -m streamlit run src/ui/streamlit_app.py
```

Open http://localhost:8501. Set `API_URL` if the API is not at
`http://127.0.0.1:8765`.

- **Sidebar:** upload a file, pick a dataset (shown with upload time and row
  count; failed uploads are marked so they can be deleted), switch chats.
  *Delete chats* removes one, several or all chats at once; deleting chats
  or a dataset asks for a confirmation tick first.
- **Ask tab:** each answer shows the summary, a caveat box, the chart and the
  table. CSV / Excel files are prepared only when you click
  *Prepare downloads*. Reopened chats show the stored summary and chart.
  Ask "what columns are there?" to see every column with its type and an
  example, plus suggested questions you can click to ask.
- **Data tab:** what the cleaning changed and flagged, the column profile and
  the schema the model sees.

**3. Postman**

Import `postman/Text-to-SQL.postman_collection.json` (and optionally the
`LocalAnalyst - local` environment in `postman/`). Its 25 requests run in the
order the app uses them: health → upload `data/input_data/messy_sales.csv` →
wrong file type → list → details → first look → cleaning review → apply the
suggestions → download the cleaned table → ask → follow-up → chart change →
top 5 rows → open-ended message → columns → unanswerable → write request →
prompt injection → chats → downloads → delete chat → delete all chats →
delete dataset. Requests save `datasetId`, `chatId` and `lastSql` for the ones
after them, and every request has tests. Set Postman's working directory
(Settings → General) to the project folder so the upload finds its file.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check and model in use |
| POST | `/datasets` | Upload a file → clean → SQLite → profile (201) |
| GET | `/datasets` | List datasets, newest first |
| GET | `/datasets/{dataset_id}` | Status, cleaning report, schema, profile (`?include_profile=false` to skip) |
| GET | `/datasets/{dataset_id}/preview?rows=5` | First rows of each table, a short description and starter questions (no model call) |
| GET | `/datasets/{dataset_id}/review` | Cleaning decisions to review: options, suggested pick, last choice |
| POST | `/datasets/{dataset_id}/review` | Apply `{"choices": {table: {column: {issue: option}}}}`, `{"use_suggestions": true}` or `{"skip": true}`; re-cleans the original file |
| DELETE | `/datasets/{dataset_id}` | Delete a dataset |
| POST | `/datasets/{dataset_id}/ask` | `{"question", "chat_id"?, "include_figures"?}` → SQL, rows, chart, summary, caveats, attempts |
| GET | `/datasets/{dataset_id}/chats` | List chats, most recent first |
| GET | `/datasets/{dataset_id}/chats/{chat_id}?include_figures=` | Chat turns with 20-row previews and the stored summary, caveats, chart spec (and Plotly figures unless `include_figures=false`) |
| DELETE | `/datasets/{dataset_id}/chats/{chat_id}` | Delete a chat |
| DELETE | `/datasets/{dataset_id}/chats?chat_ids=1&chat_ids=2` or `?all=true` | Delete several chats at once (nothing is deleted if an id is unknown) |
| GET | `/datasets/{dataset_id}/chats/{chat_id}/turns/{position}/download?format=csv\|xlsx` | Full result as CSV or Excel |

Status codes on `/ask`: **200** answered / empty / schema / refused / failed,
**400** blocked by the question guardrail, **404** unknown dataset or chat,
**409** dataset not ready, **422** missing question, **503** model unreachable
or too slow.

A `schema` answer has `sql: null`, the column table in `result`
(`table, column, type, example, missing`), a one-sentence `summary` and
`suggestions` (questions to try). Its download returns that column table as
CSV / Excel; refused, failed and blocked turns have no download (409).

Example:

```bash
curl -X POST http://127.0.0.1:8765/datasets/DS_20261116_101500_AB12/ask -H "Content-Type: application/json" -d "{\"question\": \"What are total sales by region?\"}"
```

## Evaluation

Run against held-out data that was written after tuning and never used to
tune (30 questions over 4 datasets, 5 follow-up conversations, 10 hostile
prompts, 2 new messy files with an answer key):

```bash
python -m src.evaluation.run_evaluation
```

Add `--skip-cpu` to skip the CPU-only latency run. Reports are written to
`data/output_data/evaluation/latest_run/`.

| Metric | qwen2.5-coder:7b (GPU) | qwen2.5:7b-instruct (GPU, default) | qwen2.5:3b (GPU) |
|---|---|---|---|
| Held-out questions correct | 29/30 | 29/30 | 25/30 |
| Follow-up conversations correct | 5/5 | 4/5 | 4/5 |
| Chart type correct | 18/18 | 18/18 | 17/17 |
| Normal questions wrongly blocked | 0 | 0 | 0 |
| Unsafe outcomes | 0 | 0 | 0 |
| Write requests refused | 4/4 | 3/4 | 3/4 |
| Average seconds per SQL answer | 3.2 | 3.2 | 3.4 |

Measured 19 Nov 2026. Seconds count the SQL step only; in the app a full answer
(rewrite, SQL, summary) takes about 7-15 s on the GPU profile.

The 3B on CPU alone answered 6/10 correctly at 6.4 s per answer. Cleaning
passes 28/28 checks on the held-out messy files. That score is somewhat
optimistic, because two cleaning rules were fixed after the first run on those
same files (it scored 25/28 before).

**Stress set** (`data/input_data/eval/stress_questions.json`): 57 wordings small
models tend to get wrong (count vs sum, average vs total, ranking, dates,
partial names, multi-value columns, percentages, comparisons, top per group),
graded the same way. qwen2.5:7b answers 56-57/57 (it varies by about one
question between runs), qwen2.5-coder:7b 55/57, qwen2.5:3b 42/57. Run it for any installed model:

```bash
python -m src.evaluation.run_stress qwen2.5:7b-instruct qwen2.5:3b-instruct
```

A query counts as correct when its result matches the expected rows, whatever
the column names or row order. "Unsafe" means the data changed, or an answered
query wrote data or read system or raw tables.

**Known weak spots:** ambiguous measures ("which product sells best": units or sales?) are chosen
without saying so; and "average salary of employees with a known salary"
still differs from the expected answer for both 7B models.

## Tests

```bash
python -m pytest
```

284 tests cover cleaning, storage, guardrails, charts, summaries, memory, the
pipeline and every API route. They need no model: the model calls are
scripted.

## Notes

- Each upload gets its own folder in `datasets/` with the SQLite database,
  cleaning report, profile, schema text, chat history and a trace log of every
  question.
- See `project_architecture.txt` for the full design notes and cleaning rules.
