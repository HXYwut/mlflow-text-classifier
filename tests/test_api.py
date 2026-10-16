"""API tests. The model is monkeypatched so the suite runs in CI
without needing a trained artifact or downloaded data."""

from fastapi.testclient import TestClient

from app.main import app


def _fake_predict(text: str):
    # Stub: route on a keyword instead of a real model
    if any(w in text.lower() for w in ["hockey", "puck", "nhl", "goal", "ice"]):
        return "hockey", 0.92
    if any(w in text.lower() for w in ["space", "orbit", "rocket", "nasa", "moon"]):
        return "space", 0.88
    return "hockey", 0.51  # default


client = TestClient(app)
app.dependency_overrides = {}  # no FastAPI deps to override; we monkeypatch below


def setup_function():
    import app.main as m

    m.predict = _fake_predict


# ---------- /health ----------

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------- /predict: happy path ----------

def test_predict_hockey_text():
    r = client.post("/predict", json={"text": "The puck slid past the goalie into the net."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "hockey"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_space_text():
    r = client.post("/predict", json={"text": "NASA launched a new rocket into orbit around the Moon."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "space"
    assert 0.0 <= body["probability"] <= 1.0


# ---------- /predict: edge cases ----------

def test_predict_empty_string_rejected():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length


def test_predict_whitespace_only_rejected():
    # pydantic passes (length > 0), but our handler strips -> empty -> 422
    r = client.post("/predict", json={"text": "   \n\t  "})
    assert r.status_code == 422


def test_predict_missing_text_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_special_characters_and_emoji():
    r = client.post("/predict", json={"text": "Goal!!! 🏒 #hockey @playoffs"})
    assert r.status_code == 200
    assert r.json()["label"] == "hockey"


def test_predict_long_text_accepted():
    long_text = "hockey " * 3000  # ~21k chars, within 20k? bump
    long_text = long_text[:19_000]
    r = client.post("/predict", json={"text": long_text})
    assert r.status_code == 200
