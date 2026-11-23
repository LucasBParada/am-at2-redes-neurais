import pytest
from fastapi.testclient import TestClient

from src.app import app


VALID_PAYLOAD = {
    "marca": "Ford",
    "quilometragem": 80000,
    "ano_registro": 2012,
    "numero_donos": 2,
    "combustivel": "Petrol",
    "carroceria": "Hatchback",
    "motor_litros": 1.6,
    "cambio": "Manual",
    "portas": 5,
    "assentos": 5,
    "classe_emissao": "Euro 5",
    "historico_completo": 0,
}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_diagnostico_modelo(client):
    response = client.get(
        "/diagnostico/modelo"
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


def test_predict_com_entrada_valida(client):
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["categoria_prevista"] in {
        "Economico",
        "Intermediario",
        "Premium",
    }

    assert 0 <= body["score_confianca"] <= 1


def test_predict_rejeita_tipo_invalido(client):
    payload = {
        **VALID_PAYLOAD,
        "quilometragem": "muitos quilometros",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_rejeita_categoria_invalida(client):
    payload = {
        **VALID_PAYLOAD,
        "combustivel": "Agua",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_aceita_numero_de_donos_ausente(client):
    payload = {
        key: value
        for key, value in VALID_PAYLOAD.items()
        if key != "numero_donos"
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200