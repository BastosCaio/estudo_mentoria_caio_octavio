# priority_classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Estudo e preparação para um projeto de data science de classificação de prioridade de tickets de suporte técnico.

O projeto começou analisando outro dataset (`customer_support_tickets.csv`, sob o pacote `customer_support_analytics`), mas esse dataset se mostrou inadequado para o objetivo da análise. Esse trabalho — e um segundo dataset candidato, também descartado — foi arquivado em `discarted_datasets/` e não faz parte do pipeline ativo. O trabalho ativo está em `priority_classification/`, usando o dataset Kaggle "Technical Support Dataset".

## Project Organization

```
├── LICENSE                      <- Open-source license if one is chosen
├── Makefile                     <- Makefile with convenience commands like `make lint` or `make test`
├── README.md                    <- The top-level README for developers using this project.
├── pyproject.toml                <- Project configuration file with package metadata for
│                                    priority_classification and configuration for tools like ruff
├── uv.lock                      <- Locked dependency versions, managed by uv
│
├── discarted_datasets/          <- Datasets e pipeline explorados e descartados; só referência
│   │                                histórica — não fazem parte do pyproject.toml, do lint nem de
│   │                                nenhum target do Makefile.
│   ├── customer_support_analytics/   <- Pacote cookiecutter original + notebooks de EDA de
│   │                                    customer_support_tickets.csv (dataset descartado)
│   └── ticket_helpdesk_multilingual/ <- EDA de um segundo dataset candidato, também descartado
│
└── priority_classification/     <- Projeto ativo: classificação de prioridade de tickets de
    │                                suporte técnico. Fonte de código para uso neste projeto.
    ├── __init__.py              <- Torna priority_classification um módulo Python instalável
    │
    ├── data
    │   ├── raw                 <- Technical Support Dataset.csv (o dado original, imutável)
    │   └── processed           <- train.csv / val.csv / test.csv, gerados por modeling/dataset.py
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── config.py           <- Paths do projeto (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    │   │                          ...) relativos a PROJ_ROOT, e configuração do loguru
    │   └── dataset.py          <- CLI Typer: split treino/val/teste (70/15/15), agrupado por
    │                              `Ticket ID` e estratificado por `Priority`
    │
    └── notebooks                <- Jupyter notebooks. Convenção de nome: número (ordenação),
                                     iniciais do autor e descrição curta separada por `-`.
        └── cb-eda-technical-support-dataset.ipynb  <- EDA que avalia a qualidade do dataset
                                     (nulos, duplicatas de categoria, consistência temporal, SLA,
                                     geolocalização) antes de decidir usá-lo para modelagem.
```

--------
