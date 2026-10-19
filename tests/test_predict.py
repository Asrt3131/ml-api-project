"""Tests for prediction endpoints."""


def test_predict_valid(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1, 2)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["class_name"] in ("setosa", "versicolor", "virginica")


def test_predict_setosa(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json()["prediction"] == 0


def test_predict_virginica(client):
    r = client.post("/predict", json={"features": [6.3, 3.3, 6.0, 2.5]})
    assert r.status_code == 200
    assert r.json()["prediction"] == 2


def test_predict_wrong_length(client):
    r = client.post("/predict", json={"features": [5.1, 3.5]})
    assert r.status_code == 422


def test_predict_too_many_features(client):
    r = client.post("/predict", json={"features": [1, 2, 3, 4, 5]})
    assert r.status_code == 422


def test_predict_missing_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post(
        "/predict/batch",
        json={
            "samples": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [6.3, 3.3, 6.0, 2.5]},
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty(client):
    r = client.post("/predict/batch", json={"samples": []})
    assert r.status_code == 422