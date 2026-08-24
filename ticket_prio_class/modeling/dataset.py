from pathlib import Path
import shutil

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import kagglehub
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

app = typer.Typer()

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

KAGGLE_DATASET = "albertobircoci/support-ticket-priority-dataset-50k"


@app.command()
def get_dataset(raw_data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Baixa a versão mais recente do dataset via kagglehub e copia o(s) CSV(s) para
    `raw_data_dir`, para que `main` (abaixo) tenha o que ler.

    O download em si é exatamente o pedido pelo usuário:

        import kagglehub

        # Download latest version
        path = kagglehub.dataset_download("albertobircoci/support-ticket-priority-dataset-50k")

        print("Path to dataset files:", path)

    `kagglehub` baixa para o próprio cache dele (`~/.cache/kagglehub/...`), fora do
    repositório — a cópia para `raw_data_dir` é um passo adicional para deixar o
    arquivo onde `main` espera encontrá-lo, seguindo o mesmo layout de
    `priority_classification` (dataset bruto em `data/raw/`).
    """
    # Download latest version
    path = kagglehub.dataset_download(KAGGLE_DATASET)

    print("Path to dataset files:", path)

    downloaded_dir = Path(path)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for csv_path in sorted(downloaded_dir.glob("*.csv")):
        destination = raw_data_dir / csv_path.name
        shutil.copy2(csv_path, destination)
        copied.append(destination)
        logger.success(f"Copiado para {destination}")

    if not copied:
        logger.warning(
            f"Nenhum .csv encontrado em {downloaded_dir}; nada foi copiado para {raw_data_dir}"
        )

    return copied


def _group_level_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test without splitting any `group_col` value
    across sets (e.g. no company appears in more than one split), while
    approximately stratifying by `stratify_col`.

    Because a group can only be assigned as a whole, stratification happens
    at the group level: each group is labeled with its own most frequent
    `stratify_col` value, and that pseudo-label drives the stratified split.

    With few groups and a rare dominant label (this dataset: 25 companies,
    only 2 with `priority` mode "high"), the second split (val vs. test)
    can end up with too few groups per label to stratify — in that case this
    falls back to a plain random split for that step and logs a warning,
    instead of raising.
    """
    group_label = (
        df.groupby(group_col)[stratify_col]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("group_label")
    )
    groups = group_label.index.to_numpy()
    labels = group_label.to_numpy()

    train_groups, holdout_groups, _, holdout_labels = train_test_split(
        groups,
        labels,
        train_size=TRAIN_SIZE,
        stratify=labels,
        random_state=random_state,
    )
    val_fraction_of_holdout = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    try:
        val_groups, test_groups = train_test_split(
            holdout_groups,
            train_size=val_fraction_of_holdout,
            stratify=holdout_labels,
            random_state=random_state,
        )
    except ValueError as exc:
        logger.warning(
            f"Não foi possível estratificar o split val/test por '{stratify_col}' "
            f"a nível de grupo ({exc}); caindo para split aleatório (sem "
            "estratificação) nesse passo — esperado quando sobram poucos grupos "
            "por classe depois do split de treino."
        )
        val_groups, test_groups = train_test_split(
            holdout_groups, train_size=val_fraction_of_holdout, random_state=random_state
        )

    train_df = df[df[group_col].isin(train_groups)]
    val_df = df[df[group_col].isin(val_groups)]
    test_df = df[df[group_col].isin(test_groups)]
    return train_df, val_df, test_df


def _log_split_summary(
    name: str, split_df: pd.DataFrame, total_len: int, stratify_col: str
) -> None:
    pct = len(split_df) / total_len * 100
    dist = (split_df[stratify_col].value_counts(normalize=True) * 100).round(1).to_dict()
    logger.info(f"{name}: {len(split_df)} tickets ({pct:.1f}%) | {stratify_col} dist: {dist}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "Support_tickets.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    group_col: str = "company_id",
    stratify_col: str = "priority",
    random_state: int = RANDOM_STATE,
):
    """Split Support_tickets.csv (baixado via `get_dataset`, acima) em train/val/test (70/15/15).

    Diferente de `priority_classification` (cujo `Ticket ID` é único por linha, sem
    entidade repetida), este dataset tem `company_id`: só 25 empresas, cada uma com
    ~2.000 chamados. Sem agrupar por `company_id`, um split por linha deixaria
    chamados da mesma empresa em treino e teste ao mesmo tempo — vazamento real, já
    que a mistura de `priority` varia muito entre empresas (de 0% a 55% de `high`).
    Por isso o split aqui é por grupo de fato (não degenera em split simples como no
    dataset irmão).
    """
    logger.info(f"Lendo dataset bruto de {input_path}")
    df = pd.read_csv(input_path)

    logger.info(
        f"Dividindo {len(df)} tickets agrupando por '{group_col}' "
        f"e estratificando por '{stratify_col}' (train={TRAIN_SIZE:.0%}, "
        f"val={VAL_SIZE:.0%}, test={TEST_SIZE:.0%}, random_state={random_state})"
    )
    train_df, val_df, test_df = _group_level_split(df, group_col, stratify_col, random_state)

    # Sanidade: toda linha foi atribuída a exatamente um split, e nenhum grupo
    # (empresa) aparece em mais de um split.
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    total = len(df)
    _log_split_summary("train", train_df, total, stratify_col)
    _log_split_summary("val", val_df, total, stratify_col)
    _log_split_summary("test", test_df, total, stratify_col)
    logger.success(f"Split salvo em {output_dir} (train.csv, val.csv, test.csv)")


if __name__ == "__main__":
    app()
