from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_file_found"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "thresholds" in r.json()


def test_predict_returns_valid_result():
    r = client.post("/predict", json={"url": "https://www.bbc.co.uk/news"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["verdict"] in {"legitimate", "suspicious", "phishing"}


def test_predict_flags_obvious_phishing():
    r = client.post("/predict", json={"url": "http://paypal-secure-login.verify-account.weeblysite.com/signin"})
    assert r.json()["verdict"] == "phishing"


def test_predict_rejects_empty_url():
    assert client.post("/predict", json={"url": ""}).status_code == 422


def test_batch():
    r = client.post("/predict/batch", json={"urls": ["https://github.com", "http://192.168.1.5/login"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_prediction_is_logged_and_feedback_accepted():
    r = client.post("/predict", json={"url": "https://www.bbc.co.uk/news"})
    rid = r.json()["request_id"]
    fb = client.post("/feedback", json={"request_id": rid, "true_label": 0})
    assert fb.status_code == 200
    stats = client.get("/stats").json()
    assert stats["total_predictions"] >= 1
    assert stats["labeled"] >= 1


def test_feedback_unknown_request_id():
    r = client.post("/feedback", json={"request_id": "does-not-exist", "true_label": 1})
    assert r.status_code == 404


def test_feedback_rejects_bad_label():
    r = client.post("/feedback", json={"request_id": "x", "true_label": 2})
    assert r.status_code == 422


def test_batch_is_logged():
    before = client.get("/stats").json()["total_predictions"]
    client.post("/predict/batch", json={"urls": ["https://github.com", "http://192.168.1.5/login"]})
    after = client.get("/stats").json()["total_predictions"]
    assert after == before + 2