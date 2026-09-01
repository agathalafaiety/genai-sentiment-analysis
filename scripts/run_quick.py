"""Executa preparacao, treinamento e avaliacao classica no modo rapido."""

from sentiment_analysis.data import prepare_dataset
from sentiment_analysis.training import train_classical_models


def main() -> None:
    paths = prepare_dataset("quick")
    import pandas as pd

    splits = {name: pd.read_csv(path) for name, path in paths.items()}
    metadata = train_classical_models(splits, mode="quick")
    print(f"Melhor modelo: {metadata['best_model']}")


if __name__ == "__main__":
    main()
