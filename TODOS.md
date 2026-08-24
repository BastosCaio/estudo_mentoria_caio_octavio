# TODOS

## Documentation

### Document `ticket_prio_class/` in CLAUDE.md, pyproject.toml, and Makefile

**What:** `CLAUDE.md`, `pyproject.toml`'s `[project]` name/description, and `Makefile`'s
`PROJECT_NAME` only ever describe `priority_classification` (the sibling dataset that
turned out to have no real modeling signal). `ticket_prio_class/` — a second, fully
functional package with a working EDA notebook and a working Random Forest classifier
(test macro-F1 0.910) — isn't mentioned anywhere in these files.

**Why:** A future session (human or Claude) that trusts `CLAUDE.md` alone will miss the
package that actually works and reason from stale assumptions — this already happened
once, in the `/office-hours` session that produced the rare-class-audit design doc, and
had to be corrected mid-conversation.

**Context:** Discovered during `/office-hours` (2026-08-24) while designing the
rare-class recall audit (`ticket_prio_class/docs/designs/rare-class-recall-audit.md`).
`ticket_prio_class/modeling/dataset.py` and `config.py` already exist and follow the
same Typer `app()` + `config.py`-constants pattern as `priority_classification`;
`ticket_prio_class/notebooks/1.0-eda-support-ticket-priority-dataset.ipynb` and
`2.0-classificacao-prioridade.ipynb` document the EDA and modeling work respectively.
Fix should update: `CLAUDE.md`'s Project/Architecture sections, `pyproject.toml`'s
`[project]` `name`/`description` (or note that the project is now two packages), and
`Makefile`'s `PROJECT_NAME`.

**Effort:** S
**Priority:** P3
**Depends on:** None
