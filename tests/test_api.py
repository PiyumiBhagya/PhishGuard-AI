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