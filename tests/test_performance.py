"""RNF10 - Desempenho da API de predição.

A API deve responder a uma requisição de predição em até 200 ms no percentil 95,
medido em 200 requisições sequenciais com o artefato já carregado, em ambiente
local de referência (execução em processo, sem latência de rede).

Dois testes:
- test_predict_latency_p95_artefato_de_teste: usa o artefato sintético do conftest
  (roda sempre, inclusive em CI).
- test_predict_latency_p95_artefato_real: usa models/fraud_model_artifact.joblib
  (modelo treinado no PaySim); é pulado se o arquivo não existir.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import BASE_DIR, settings
from app.main import app
from app.services.model_service import get_predictor

N_REQUISICOES = 200
LIMITE_P95_MS = 200.0
ARTEFATO_REAL = BASE_DIR / "models" / "fraud_model_artifact.joblib"


def _medir_p95(client: TestClient, payload: dict) -> float:
    # Aquecimento: carrega o artefato em memória antes da medição (pré-condição do RNF10).
    resposta = client.post("/api/v1/predict", json=payload)
    assert resposta.status_code == 200, resposta.text
    tempos_ms = []
    for _ in range(N_REQUISICOES):
        inicio = time.perf_counter()
        resposta = client.post("/api/v1/predict", json=payload)
        tempos_ms.append((time.perf_counter() - inicio) * 1000)
        assert resposta.status_code == 200
    return float(np.percentile(tempos_ms, 95))


def test_predict_latency_p95_artefato_de_teste():
    client = TestClient(app)
    p95 = _medir_p95(client, {"features": {"feature_1": 0.95, "feature_2": 1}})
    print(f"\nRNF10 (artefato de teste): p95 = {p95:.1f} ms em {N_REQUISICOES} requisições")
    assert p95 <= LIMITE_P95_MS


@pytest.mark.skipif(not Path(ARTEFATO_REAL).exists(), reason="artefato real não encontrado")
def test_predict_latency_p95_artefato_real(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "model_artifact_path", ARTEFATO_REAL)
    get_predictor.cache_clear()
    payload = {"features": {
        "step": 1, "type": "TRANSFER", "amount": 181.0,
        "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFlaggedFraud": 0,
    }}
    client = TestClient(app)
    p95 = _medir_p95(client, payload)
    print(f"\nRNF10 (artefato real): p95 = {p95:.1f} ms em {N_REQUISICOES} requisições")
    assert p95 <= LIMITE_P95_MS
