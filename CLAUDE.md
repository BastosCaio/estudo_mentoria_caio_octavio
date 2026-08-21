# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Data science study/prep project (`customer_support_analytics`) analyzing the Kaggle "Customer Support Ticket Dataset" (`data/raw/customer_support_tickets.csv`). Scaffolded from cookiecutter-data-science. Notebooks and docstrings are written in Portuguese (pt-BR); code identifiers are English.

## Commands

Dependency management is via `uv`, not pip directly.

```
make requirements   # uv sync — install/update dependencies
make lint           # ruff format --check && ruff check
make format         # ruff check --fix && ruff format (run before committing)
make test           # python -m pytest tests
make data           # runs customer_support_analytics/dataset.py (raw -> processed split)
make clean          # remove __pycache__ / *.pyc
```

Run a single test: `python -m pytest tests/test_data.py::test_code_is_tested`

Module entry points are Typer CLIs, invoked directly, e.g.:
```
python customer_support_analytics/dataset.py --input-path data/raw/customer_support_tickets.csv --output-dir data/processed
```

## Architecture

- **`customer_support_analytics/config.py`** — loads `.env` via `python-dotenv` and defines all project paths (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, `FIGURES_DIR`, etc.) relative to `PROJ_ROOT`. Every script imports paths from here rather than hardcoding them, and configures `loguru` to play nicely with `tqdm` progress bars.
- **`customer_support_analytics/dataset.py`** — the one fully-implemented pipeline stage. Splits the raw CSV into `train.csv`/`val.csv`/`test.csv` (70/15/15) in `data/processed/`. The split is **group-level by `Customer Email`** (a customer's tickets never span more than one split, avoiding leakage) and **approximately stratified by `Ticket Priority`** — since stratification can't be exact at the row level once grouping is enforced, each customer group is first assigned a pseudo-label (its most frequent `Ticket Priority`) and that label drives `train_test_split`. Run via its Typer `app()`.
- **`customer_support_analytics/features.py`, `plots.py`, `modeling/train.py`, `modeling/predict.py`** — still cookiecutter placeholder stubs (unmodified `# ---- REPLACE ... ----` boilerplate with a fake `tqdm` loop). Treat these as scaffolding to be filled in, not as a real contract to preserve.
- **`tests/test_data.py`** — currently a placeholder (`assert False`), not a real test suite yet.
- **`notebooks/`** — exploratory analysis, numbered per cookiecutter convention (`<n>.<owner-initials>-<description>.ipynb`), split by which data stage they read: `1.0-cb-eda-raw-pre-split.ipynb` runs on `data/raw/customer_support_tickets.csv` and only checks what's needed to justify the split parameters in `dataset.py` (customer-email repetition, `Ticket Priority` distribution); `1.1-cb-eda-train.ipynb` runs on `data/processed/train.csv` (requires `make data` to have been run first) and does the deeper EDA (8,469-row full dataset structurally: missing values concentrated in `Resolution`, `First Response Time`, `Time to Resolution`, `Customer Satisfaction Rating` — these track open/unresolved tickets rather than being random gaps). Deliberately reading only the train split — not val/test — avoids leaking those into modeling decisions.

## Conventions

- Ruff is configured with `line-length = 99` and import sorting (`extend-select = ["I"]`), first-party imports (`customer_support_analytics`) sorted separately and force-sorted within sections — always run `make format` rather than hand-ordering imports.
- New pipeline scripts follow the existing Typer `app()` + `@app.command() def main(...)` pattern, with I/O paths as parameters defaulting to the `config.py` directory constants.
