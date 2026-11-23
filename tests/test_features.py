import pandas as pd

from src.features import (
    FEATURES,
    clean_and_label,
    model_input,
    project_root,
)


def test_clean_and_label_remove_duplicados_e_cria_alvo():
    raw = pd.read_csv(
        project_root()
        / "data"
        / "dataset.csv"
    )

    prepared = clean_and_label(raw)

    assert len(prepared) == 2849

    assert set(
        prepared["faixa_preco"].astype(str)
    ) == {
        "Economico",
        "Intermediario",
        "Premium",
    }

    assert set(FEATURES).issubset(
        prepared.columns
    )


def test_model_input_mantem_ordem_das_features():
    data = pd.DataFrame(
        [
            {
                feature: None
                for feature in FEATURES
            }
        ]
    )

    assert list(
        model_input(data).columns
    ) == FEATURES