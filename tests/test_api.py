import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if not os.path.exists(os.path.join(ROOT, "models", "crop_model.pkl")):
    pytest.skip("Model not trained yet - run `python train_model.py` first.", allow_module_level=True)

from app import create_app  # noqa: E402

RICE_INPUT = dict(N=90, P=42, K=43, temperature=20.9, humidity=82.0, ph=6.5, rainfall=202.9)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok", "model_loaded": True}


def test_crops(client):
    data = client.get("/api/crops").get_json()
    assert data["count"] == 22
    assert "rice" in data["crops"]


def test_model_info(client):
    data = client.get("/api/model-info").get_json()
    assert data["test_accuracy"] > 0.95
    assert set(data["features"]) == set(RICE_INPUT)
    assert "RandomForest" in data["model_comparison"]


def test_recommend_rice(client):
    res = client.post("/api/recommend", json=RICE_INPUT)
    assert res.status_code == 200
    data = res.get_json()
    recs = data["recommendations"]
    assert len(recs) == 3
    assert recs[0]["crop"] == "rice"
    for rec in recs:
        assert set(rec) == {"crop", "confidence_pct", "explanation", "warnings"}
        assert 0 <= rec["confidence_pct"] <= 100
        assert len(rec["explanation"]) == 3
    # Sorted by confidence, highest first
    confidences = [r["confidence_pct"] for r in recs]
    assert confidences == sorted(confidences, reverse=True)
    assert data["notes"] == []


def test_recommend_accepts_numeric_strings(client):
    payload = {k: str(v) for k, v in RICE_INPUT.items()}
    res = client.post("/api/recommend", json=payload)
    assert res.status_code == 200


def test_recommend_top_k(client):
    res = client.post("/api/recommend", json={**RICE_INPUT, "top_k": 5})
    assert len(res.get_json()["recommendations"]) == 5


@pytest.mark.parametrize("top_k", [0, 11, "3", 2.5, True])
def test_recommend_bad_top_k(client, top_k):
    res = client.post("/api/recommend", json={**RICE_INPUT, "top_k": top_k})
    assert res.status_code == 400
    assert "top_k" in res.get_json()["details"]


def test_recommend_missing_fields(client):
    res = client.post("/api/recommend", json={"N": 90})
    assert res.status_code == 400
    details = res.get_json()["details"]
    assert set(details) == {"P", "K", "temperature", "humidity", "ph", "rainfall"}


@pytest.mark.parametrize(
    "field,value",
    [("ph", 15), ("humidity", -5), ("N", "abc"), ("rainfall", True), ("K", None)],
)
def test_recommend_invalid_values(client, field, value):
    res = client.post("/api/recommend", json={**RICE_INPUT, field: value})
    assert res.status_code == 400
    assert field in res.get_json()["details"]


def test_recommend_non_json_body(client):
    res = client.post("/api/recommend", data="hello", content_type="text/plain")
    assert res.status_code == 400


def test_recommend_warns_outside_training_range(client):
    # N=250 is a valid value but far above anything in the training data (max 140)
    res = client.post("/api/recommend", json={**RICE_INPUT, "N": 250})
    assert res.status_code == 200
    notes = res.get_json()["notes"]
    assert len(notes) == 1 and notes[0].startswith("N = 250")


def test_unknown_route_returns_json(client):
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.get_json() == {"error": "Not found"}


def test_cors_header(client):
    res = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    # flask-cors echoes the caller's origin back when all origins are allowed
    assert res.headers.get("Access-Control-Allow-Origin") in ("*", "http://localhost:3000")


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"AgriSense" in res.data
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
