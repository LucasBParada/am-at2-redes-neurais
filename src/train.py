"""Treina e salva o pipeline de classificação.

Uso:

    uv run python -m src.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.features import (
    CATEGORICAS,
    NUMERICAS,
    clean_and_label,
    model_input,
    project_root,
)


SEED = 77


def build_pipeline() -> Pipeline:
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),
                NUMERICAS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CATEGORICAS,
            ),
        ]
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        alpha=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
    )

    return Pipeline(
        [
            (
                "preprocessamento",
                preprocessador,
            ),
            (
                "mlp",
                mlp,
            ),
        ]
    )


def train(
    data_path: Path,
    model_path: Path,
) -> dict[str, float | int]:
    raw = pd.read_csv(data_path)

    data = clean_and_label(raw)

    X = model_input(data)
    y = data["faixa_preco"].astype(str)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED,
        )
    )

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "rows": int(len(data)),
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_test,
                predictions,
            )
        ),
        "iterations": int(
            pipeline.named_steps[
                "mlp"
            ].n_iter_
        ),
    }

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    root = project_root()

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            root
            / "data"
            / "dataset.csv"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            root
            / "models"
            / "pipeline_rede_neural.joblib"
        ),
    )

    args = parser.parse_args()

    metrics = train(
        args.data,
        args.model,
    )

    print(
        f"Modelo salvo em: "
        f"{args.model.resolve()}"
    )
    print(
        f"Linhas usadas: {metrics['rows']}"
    )
    print(
        f"Acurácia: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"MCC: {metrics['mcc']:.4f}"
    )
    print(
        f"Iterações: {metrics['iterations']}"
    )


if __name__ == "__main__":
    main()