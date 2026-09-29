from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)
API = "/api/v1"


def test_health():
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "nexus-api", "version": "1.0.0"}
    assert r.headers["X-Request-ID"].startswith("req_")


def test_validation_error_uses_contract():
    r = client.post(f"{API}/chat", json={"message": ""})
    body = r.json()
    assert r.status_code == 422
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["meta"]["request_id"] == r.headers["X-Request-ID"]


def test_unknown_model_rejected():
    assert client.post(f"{API}/chat", json={"message": "hi", "model": "gpt-2"}).status_code == 422


def test_upload_rejects_bad_file_type():
    r = client.post(f"{API}/documents/upload", files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 422
    assert "Unsupported" in r.json()["error"]["message"]


def test_upload_requires_file_or_url():
    assert client.post(f"{API}/documents/upload").status_code == 422


def test_bad_uuid_and_top_k_rejected():
    assert client.get(f"{API}/conversations/not-a-uuid").status_code == 422
    assert client.post(f"{API}/search/vector", json={"query": "x", "top_k": 99}).status_code == 422
