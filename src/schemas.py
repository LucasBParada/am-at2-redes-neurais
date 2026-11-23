from typing import Literal, Optional

from pydantic import BaseModel, Field


class CarroEntrada(BaseModel):
    """Atributos de entrada usados pelo modelo."""

    marca: str = Field(
        ...,
        min_length=2,
        max_length=30,
        description="Fabricante do veículo",
        examples=["Ford"],
    )

    quilometragem: float = Field(
        ...,
        ge=0,
        le=300_000,
        description="Milhas rodadas",
        examples=[60000],
    )

    ano_registro: int = Field(
        ...,
        ge=1990,
        le=2024,
        description="Ano de registro do veículo",
        examples=[2016],
    )

    numero_donos: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Número de donos anteriores",
        examples=[2],
    )

    combustivel: Literal[
        "Petrol",
        "Diesel",
        "Electric",
        "Petrol Hybrid",
        "Petrol Plug-in Hybrid",
        "Diesel Hybrid",
    ] = Field(
        ...,
        description="Tipo de combustível",
        examples=["Petrol"],
    )

    carroceria: Literal[
        "Hatchback",
        "SUV",
        "Saloon",
        "Estate",
        "MPV",
        "Coupe",
        "Convertible",
        "Pickup",
        "Minibus",
        "Combi Van",
    ] = Field(
        ...,
        description="Tipo de carroceria",
        examples=["Hatchback"],
    )

    motor_litros: float = Field(
        ...,
        ge=0.6,
        le=7.0,
        description="Cilindrada em litros",
        examples=[1.6],
    )

    cambio: Literal[
        "Manual",
        "Automatic",
    ] = Field(
        ...,
        description="Tipo de câmbio",
        examples=["Manual"],
    )

    portas: int = Field(
        ...,
        ge=2,
        le=5,
        description="Quantidade de portas",
        examples=[5],
    )

    assentos: int = Field(
        ...,
        ge=2,
        le=9,
        description="Quantidade de assentos",
        examples=[5],
    )

    classe_emissao: Literal[
        "Euro 1",
        "Euro 2",
        "Euro 3",
        "Euro 4",
        "Euro 5",
        "Euro 6",
    ] = Field(
        ...,
        description="Classe de emissões Euro",
        examples=["Euro 6"],
    )

    historico_completo: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "1 = histórico completo; "
            "0 = histórico não informado"
        ),
        examples=[1],
    )


class PredicaoSaida(BaseModel):
    categoria_prevista: str = Field(
        ...,
        description="Faixa de preço prevista",
        examples=["Intermediario"],
    )

    score_confianca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidade bruta da classe prevista"
        ),
        examples=[0.87],
    )