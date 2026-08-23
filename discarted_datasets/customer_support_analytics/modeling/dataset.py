from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from discarted_datasets.customer_support_analytics.modeling.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def _group_level_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test without splitting any `group_col` value
    across sets (e.g. no customer appears in more than one split), while
    approximately stratifying by `stratify_col`.

    Because a group can only be assigned as a whole, stratification happens
    at the group level: each group is labeled with its own most frequent
    `stratify_col` value, and that pseudo-label drives the stratified split.
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
    val_groups, test_groups = train_test_split(
        holdout_groups,
        train_size=val_fraction_of_holdout,
        stratify=holdout_labels,
        random_state=random_state,
    )

    train_df = df[df[group_col].isin(train_groups)]
    val_df = df[df[group_col].isin(val_groups)]
    test_df = df[df[group_col].isin(test_groups)]
    return train_df, val_df, test_df


def _log_split_summary(name: str, split_df: pd.DataFrame, total_len: int, stratify_col: str) -> None:
    pct = len(split_df) / total_len * 100
    dist = (split_df[stratify_col].value_counts(normalize=True) * 100).round(1).to_dict()
    logger.info(f"{name}: {len(split_df)} tickets ({pct:.1f}%) | {stratify_col} dist: {dist}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "customer_support_tickets.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    group_col: str = "Customer Email",
    stratify_col: str = "Ticket Priority",
    random_state: int = RANDOM_STATE,
):
    """Split customer_support_tickets.csv em train/val/test (70/15/15).

    O split é agrupado por `group_col` (nenhum cliente aparece em mais de um
    conjunto, evitando vazamento de dados) e aproximadamente estratificado
    por `stratify_col` (usando o valor mais frequente de cada grupo como
    rótulo do grupo para a estratificação).
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
    # (ex.: cliente) aparece em mais de um split.
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
