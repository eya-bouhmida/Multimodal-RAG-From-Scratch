import pymupdf
import pytest
from fastapi.testclient import TestClient

import backend.app as app_module


class FakeRetriever:
    text_chunks_count = 42
    image_chunks_count = 7

    def hybrid_search(self, query, top_k=8):
        return []

    def image_search(self, query, top_k=3):
        return []


class FakeGenerator:
    def generate_stream(self, retriever, query, history=None, uploaded_chunks=None):
        yield {"type": "token", "content": "Hello "}
        yield {"type": "token", "content": "world"}
        yield {"type": "done", "sources": [], "text_chunks": 0, "images_used": 0}


@pytest.fixture
def client():
    # No `with` block: this skips the real lifespan (model loading / Qdrant connection)
    # so state is whatever we set here instead of what lifespan() would populate.
    app_module.state["retriever"] = FakeRetriever()
    app_module.state["generator"] = FakeGenerator()
    app_module.state["uploads"] = {}
    return TestClient(app_module.app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["text_chunks_count"] == 42
    assert body["image_chunks_count"] == 7


def test_chat_streams_tokens(client):
    response = client.post("/api/chat", json={"message": "hi", "history": []})
    assert response.status_code == 200
    assert "Hello" in response.text
    assert "world" in response.text


def test_export_pdf_returns_pdf(client):
    response = client.post("/api/export-pdf", json={"title": "T", "text": "Body text"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_upload_and_delete_document(client):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox((72, 72, 500, 700), "Some content for the test upload.", fontsize=11)
    pdf_bytes = doc.tobytes()

    response = client.post(
        "/api/upload",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        data={"session_id": "test-session"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "test.pdf"
    assert body["num_chunks"] >= 1
    assert "test-session" in app_module.state["uploads"]

    delete_response = client.delete("/api/upload/test-session")
    assert delete_response.status_code == 200
    assert "test-session" not in app_module.state["uploads"]


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
        data={"session_id": "test-session"},
    )
    assert response.status_code == 400
