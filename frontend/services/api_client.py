import atexit
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

from config import (API_PORT, API_URL, BACKEND_LOG, BACKEND_START_TIMEOUT_S, BACKEND_URL, REQUEST_TIMEOUT_S,
                    ROOT)


class ApiError(Exception):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.code, self.message, self.request_id = code, message, request_id

    def __str__(self) -> str:
        return f"{self.message} ({self.code}" + (f", {self.request_id})" if self.request_id else ")")


class ApiClient:
    """The only way the frontend talks to NEXUS. Unwraps {data, meta}; raises ApiError on {error}."""

    def __init__(self, base_url: str = API_URL, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT_S)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise ApiError("BACKEND_UNREACHABLE", f"Cannot reach backend at {BACKEND_URL}: {type(e).__name__}") from e
        return _unwrap(r)

    # health / observability
    def health(self) -> dict:
        return self._request("GET", "/health")

    def metrics(self) -> dict:
        return self._request("GET", "/metrics")

    # conversations
    def conversations(self) -> list[dict]:
        return self._request("GET", "/conversations")["conversations"]

    def create_conversation(self, title: str) -> dict:
        return self._request("POST", "/conversations", json={"title": title})

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/conversations/{conversation_id}")

    def messages(self, conversation_id: str) -> list[dict]:
        return self._request("GET", f"/conversations/{conversation_id}/messages")["messages"]

    # chat
    def chat_stream(self, payload: dict) -> Iterator[tuple[str, dict]]:
        """Server-Sent Events -> (event, data). Backend errors mid-stream arrive as ('error', {...})."""
        try:
            with self.http.stream("POST", "/chat/stream", json=payload) as r:
                if r.is_error:
                    r.read()
                    _unwrap(r)
                event = "message"
                for line in r.iter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        yield event, json.loads(line[5:])
        except httpx.HTTPError as e:
            raise ApiError("BACKEND_UNREACHABLE", f"Stream interrupted: {type(e).__name__}") from e

    # documents
    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")["documents"]

    def upload_file(self, name: str, data: bytes, mime: str | None) -> dict:
        return self._request("POST", "/documents/upload", files={"file": (name, data, mime or "application/octet-stream")})

    def upload_url(self, url: str) -> dict:
        return self._request("POST", "/documents/upload", data={"url": url})

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def vector_search(self, query: str, top_k: int, rewrite: bool) -> list[dict]:
        return self._request("POST", "/search/vector", json={"query": query, "top_k": top_k, "rewrite": rewrite})["results"]

    # evaluation
    def evaluations(self, limit: int = 100) -> dict:
        return self._request("GET", "/evaluations", params={"limit": limit})


def _unwrap(r: httpx.Response) -> Any:
    if r.status_code == 204:
        return None
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.is_error:
        err = body.get("error") or {}
        raise ApiError(err.get("code", f"HTTP_{r.status_code}"), err.get("message", r.text[:200] or r.reason_phrase),
                       (body.get("meta") or {}).get("request_id"))
    return body.get("data", body)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


# ---------- backend process ----------

def _healthy() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource(show_spinner=False)
def ensure_backend() -> subprocess.Popen | None:
    """Starts `uvicorn app.main:app` once per frontend process, unless a backend already answers /health.
    The child is stopped when the frontend exits."""
    if _healthy():
        return None
    BACKEND_LOG.parent.mkdir(exist_ok=True)
    log = open(BACKEND_LOG, "ab")  # noqa: SIM115 - must outlive this function, closed with the process
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    deadline = time.monotonic() + BACKEND_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Backend exited with code {proc.returncode}. See {BACKEND_LOG}")
        if _healthy():
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Backend did not become healthy in {BACKEND_START_TIMEOUT_S}s. See {BACKEND_LOG}")
