"""Funções compartilhadas de preparação dos dados do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


NUMERICAS: Final[list[str]] = [
    "quilometragem",
    "ano_registro",
    "numero_donos",
    "motor_litros",
    "portas",
    "assentos",
    "historico_completo",
]

CATEGORICAS: Final[list[str]] = [
    "marca",
    "combustivel",
    "carroceria",
    "cambio",
    "classe_emissao",
]

FEATURES: Final[list[str]] = NUMERICAS + CATEGORICAS

CLASSES: Final[list[str]] = [
    "Economico",
    "Intermediario",
    "Premium",
]

MARCAS_COMPOSTAS: Final[dict[str, str]] = {
    "Land": "Land Rover",
    "Alfa": "Alfa Romeo",
}


def project_root() -> Path:
    """Retorna a raiz do projeto independentemente do diretório atual."""
    return Path(__file__).resolve().parents[1]


def prepare_raw_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Converte o CSV original para as colunas usadas pelo modelo."""
    required = {
        "title",
        "Price",
        "Mileage(miles)",
        "Registration_Year",
        "Previous Owners",
        "Fuel type",
        "Body type",
        "Engine",
        "Gearbox",
        "Doors",
        "Seats",
        "Emission Class",
        "Service history",
    }

    missing = required.difference(raw.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {sorted(missing)}"
        )

    result = pd.DataFrame(
        {
            "marca": (
                raw["title"]
                .astype("string")
                .str.split()
                .str[0]
                .replace(MARCAS_COMPOSTAS)
            ),
            "quilometragem": raw["Mileage(miles)"],
            "ano_registro": raw["Registration_Year"],
            "numero_donos": raw["Previous Owners"],
            "combustivel": raw["Fuel type"],
            "carroceria": raw["Body type"],
            "motor_litros": (
                raw["Engine"]
                .astype("string")
                .str.replace("L", "", regex=False)
            ),
            "cambio": raw["Gearbox"],
            "portas": raw["Doors"],
            "assentos": raw["Seats"],
            "classe_emissao": raw["Emission Class"],
            "historico_completo": (
                raw["Service history"]
                .eq("Full")
                .astype(int)
            ),
            "preco": pd.to_numeric(
                raw["Price"],
                errors="coerce",
            ),
        }
    )

    result["motor_litros"] = pd.to_numeric(
        result["motor_litros"],
        errors="coerce",
    )

    return result


def clean_and_label(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpa registros inválidos, remove duplicidades e cria o alvo."""
    result = prepare_raw_dataframe(raw)

    result = result[
        (result["quilometragem"] <= 300_000)
        & (result["ano_registro"] >= 1990)
    ]

    result = result.drop_duplicates().reset_index(
        drop=True
    )

    result["faixa_preco"] = pd.cut(
        result["preco"],
        bins=[0, 3000, 6500, np.inf],
        labels=CLASSES,
        include_lowest=True,
    )

    result = result.drop(columns="preco")

    return result


def model_input(data: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e ordena as colunas esperadas pelo pipeline."""
    missing = set(FEATURES).difference(data.columns)

    if missing:
        raise ValueError(
            "Atributos ausentes para predição: "
            f"{sorted(missing)}"
        )

    return data[FEATURES].copy()