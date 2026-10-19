"""Tests for meta endpoints."""


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 4
    assert len(body["classes"]) == 3
    assert "RandomForest" in body["model_type"]