import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"

VALID_PAYLOAD = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 50000,
    "BILL_AMT3": 50000,
    "BILL_AMT4": 50000,
    "BILL_AMT5": 50000,
    "BILL_AMT6": 50000,
    "PAY_AMT1": 0,
    "PAY_AMT2": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_predict_v1():
    payload = {**VALID_PAYLOAD, "user_id": "odd_user_1"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability_default" in data
    assert data["model_version"] in ("v1", "v2")


def test_predict_ab_routing():
    results = set()
    for uid in range(20):
        payload = {**VALID_PAYLOAD, "user_id": str(uid)}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200
        results.add(r.json()["model_version"])
    assert "v1" in results and "v2" in results, "Expected traffic to be split across both model versions"


def test_predict_no_user_id_defaults_to_v1():
    response = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_predict_missing_fields():
    response = requests.post(f"{BASE_URL}/predict", json={"LIMIT_BAL": 50000})
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_empty_data():
    response = requests.post(f"{BASE_URL}/predict", json={})
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_wrong_method():
    response = requests.get(f"{BASE_URL}/predict")
    assert response.status_code in (400, 405)
