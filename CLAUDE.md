# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Data science study/prep project focused on **priority classification of technical support tickets** (`priority_classification/`), using the Kaggle "Technical Support Dataset" (`priority_classification/data/raw/Technical Support Dataset.csv`). Scaffolded from cookiecutter-data-science. Notebooks and docstrings are written in Portuguese (pt-BR); code identifiers are English.

An earlier iteration analyzed a different dataset (`customer_support_tickets.csv`) under a `customer_support_analytics` package; that dataset turned out inadequate for the analysis goal. That package, plus a second rejected dataset candidate (a multilingual helpdesk dataset), were archived under `discarted_datasets/` — historical reference only, not wired into `pyproject.toml`, lint, or any Makefile target. Treat it as read-only history, not as current architecture.

## Commands

Dependency management is via `uv`, not pip directly.

```
make requirements   # uv sync — install/update dependencies
make lint           # ruff format --check && ruff check (targets priority_classification/)
make format         # ruff check --fix && ruff format (run before committing)
make test           # python -m pytest tests
make clean          # remove __pycache__ / *.pyc
```

`make test` currently errors — `tests/` doesn't exist (it was removed along with the earlier package). There is no test suite yet.

Module entry points are Typer CLIs, invoked directly, e.g.:
```
uv run python priority_classification/modeling/dataset.py \
  --input-path "priority_classification/data/raw/Technical Support Dataset.csv" \
  --output-dir priority_classification/data/processed
```

## Architecture

- **`priority_classification/__init__.py`** — makes the project an installable module (`pyproject.toml` project name is `priority_classification`); currently empty.
- **`priority_classification/modeling/config.py`** — loads `.env` via `python-dotenv` and defines project paths (`DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR`) relative to `PROJ_ROOT` (resolved as `priority_classification/`), and configures `loguru` to play nicely with `tqdm` progress bars.
- **`priority_classification/modeling/dataset.py`** — the one fully-implemented pipeline stage. Splits `Technical Support Dataset.csv` into `train.csv`/`val.csv`/`test.csv` (70/15/15) in `priority_classification/data/processed/`, grouping by `Ticket ID` and approximately stratifying by `Priority`. Unlike the discarded pipeline (which grouped by customer email to prevent leakage), this dataset has no customer identifier — `Ticket ID` is unique per row, so the group-level split degenerates to a plain stratified split (no leakage risk, since no entity repeats across rows); the split function is kept generic for consistency with that earlier approach. It imports `config` with a flat `from config import ...` (not package-qualified), so run it as a script (`python priority_classification/modeling/dataset.py`), which puts its own directory on `sys.path`.
- **`priority_classification/notebooks/cb-eda-technical-support-dataset.ipynb`** — EDA that vets this dataset's quality before deciding whether/how to use it: nulls are 100% structural (tied to `Status`), `Topic` has a case-duplicated category, `Agent Group`/`Support Level`/`Agent Name` are redundant with each other, a small minority of rows have chronologically-inconsistent timestamps, and `Latitude`/`Longitude` are a per-`Country` centroid rather than per-ticket geolocation.
- **No test suite currently.** `tests/` was removed along with the earlier package.
- **`discarted_datasets/`** — archived, inactive:
  - `customer_support_analytics/` — the original cookiecutter package (`config.py`, `dataset.py`, `features.py`, `plots.py`, `modeling/`) plus its EDA notebooks, built around `customer_support_tickets.csv`.
  - `ticket_helpdesk_multilingual/` — EDA of a second candidate dataset, also rejected.

## Conventions

- Ruff is configured with `line-length = 99` and import sorting (`extend-select = ["I"]`), first-party imports (`priority_classification`) sorted separately and force-sorted within sections — always run `make format` rather than hand-ordering imports.
- New pipeline scripts follow the existing Typer `app()` + `@app.command() def main(...)` pattern, with I/O paths as parameters defaulting to the `config.py` directory constants.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
