"""Download, validacao, amostragem e divisao do B2W-Reviews01."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from sentiment_analysis.config import (
    B2W_URL,
    LABELS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RATING_TO_LABEL,
    RAW_DATA_PATH,
)
from sentiment_analysis.preprocessing import combine_review_text

REQUIRED_COLUMNS = {"review_title", "review_text", "overall_rating"}
QUICK_PER_CLASS = 1_200


class DataValidationError(ValueError):
    """Indica que o dataset nao satisfaz o contrato do projeto."""


def download_dataset(destination: Path = RAW_DATA_PATH, *, force: bool = False) -> Path:
    """Baixa a revisao fixada do corpus oficial com escrita atomica."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(B2W_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary.replace(destination)
    return destination


def file_sha256(path: Path) -> str:
    """Calcula o SHA-256 em streaming para registrar proveniencia."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_data(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Colunas obrigatorias ausentes: {sorted(missing)}")
    ratings = pd.to_numeric(frame["overall_rating"], errors="coerce").dropna()
    if ratings.empty or not ratings.between(1, 5).all():
        raise DataValidationError("overall_rating deve conter apenas notas de 1 a 5")


def load_and_clean(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega apenas campos necessarios e produz texto/rotulo consistentes."""
    path = Path(path)
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=False,
    )
    validate_raw_data(frame)
    frame["rating"] = pd.to_numeric(frame["overall_rating"], errors="coerce").astype("Int64")
    frame = frame[frame["rating"].isin(RATING_TO_LABEL)].copy()
    frame["text"] = [
        combine_review_text(title, body)
        for title, body in zip(frame["review_title"], frame["review_text"], strict=True)
    ]
    frame["label"] = frame["rating"].map(RATING_TO_LABEL)
    frame = frame.loc[frame["text"].str.len() >= 3, ["text", "label", "rating"]]
    frame = frame.drop_duplicates(subset="text").reset_index(drop=True)
    validate_prepared_data(frame)
    return frame


def validate_prepared_data(frame: pd.DataFrame) -> None:
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"Dados preparados exigem colunas {sorted(required)}")
    invalid = set(frame["label"].dropna().unique()).difference(LABELS)
    if invalid:
        raise DataValidationError(f"Rotulos invalidos: {sorted(invalid)}")
    if frame["text"].isna().any() or (frame["text"].str.strip() == "").any():
        raise DataValidationError("Textos vazios nao sao permitidos")
    if set(frame["label"].unique()) != set(LABELS):
        raise DataValidationError("As tres classes precisam estar presentes")


def stratified_sample(
    frame: pd.DataFrame,
    *,
    per_class: int = QUICK_PER_CLASS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Cria amostra balanceada deterministica para o modo rapido."""
    validate_prepared_data(frame)
    pieces = []
    for label in LABELS:
        subset = frame.loc[frame["label"] == label]
        amount = min(per_class, len(subset))
        pieces.append(subset.sample(n=amount, random_state=random_state))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def split_data(
    frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Divide em treino/validacao/teste (64/16/20) de forma estratificada."""
    validate_prepared_data(frame)
    train_validation, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=random_state,
        stratify=frame["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.20,
        random_state=random_state,
        stratify=train_validation["label"],
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def prepare_dataset(
    mode: str = "quick",
    *,
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DIR,
    download: bool = True,
) -> dict[str, Path]:
    """Prepara e persiste splits portaveis para os modos quick ou full."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode deve ser 'quick' ou 'full'")
    raw_path = Path(raw_path)
    if download and not raw_path.exists():
        download_dataset(raw_path)
    frame = load_and_clean(raw_path)
    if mode == "quick":
        frame = stratified_sample(frame)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_frame in split_data(frame).items():
        path = output_dir / f"{mode}_{split_name}.csv"
        split_frame.to_csv(path, index=False)
        paths[split_name] = path
    return paths


def load_splits(mode: str = "quick", directory: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    paths = {
        name: Path(directory) / f"{mode}_{name}.csv" for name in ("train", "validation", "test")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Splits ausentes: {missing}. Execute sentiment-data --mode {mode}."
        )
    splits = {name: pd.read_csv(path) for name, path in paths.items()}
    for split in splits.values():
        validate_prepared_data(split)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.force_download:
        download_dataset(force=True)
    paths = prepare_dataset(args.mode)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
