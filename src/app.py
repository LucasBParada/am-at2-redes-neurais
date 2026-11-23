from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from src.features import model_input
from src.schemas import CarroEntrada, PredicaoSaida


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "pipeline_rede_neural.joblib"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    app.state.modelo = joblib.load(MODEL_PATH)
    app.state.model_path = MODEL_PATH

    yield

    app.state.modelo = None


app = FastAPI(
    title=(
        "API de Faixa de Preço "
        "de Carros Usados (UK)"
    ),
    description=(
        "Classifica um carro usado em "
        "Economico, Intermediario ou Premium "
        "usando uma MLPClassifier. "
        "O score é a probabilidade bruta "
        "do modelo e não uma probabilidade "
        "calibrada."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo")
def diagnostico_modelo():
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


@app.post(
    "/predict",
    response_model=PredicaoSaida,
    status_code=200,
)
def predict(
    carro: CarroEntrada,
    request: Request,
):
    modelo = request.app.state.modelo

    entrada = model_input(
        pd.DataFrame(
            [carro.model_dump()]
        )
    )

    probabilidades = modelo.predict_proba(
        entrada
    )[0]

    indice = int(
        probabilidades.argmax()
    )

    return PredicaoSaida(
        categoria_prevista=str(
            modelo.classes_[indice]
        ),
        score_confianca=float(
            probabilidades[indice]
        ),
    )