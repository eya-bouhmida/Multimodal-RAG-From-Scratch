import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.schemas import ChatRequest, ExportPdfRequest, HealthResponse, UploadResponse
from src.generator import MedLensGenerator
from src.parser import parse_pdf_to_chunks
from src.pdf_export import build_pdf
from src.retriever import HybridRetriever, select_relevant_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medlens")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_CHUNKS = 120

# Curated subset (whitelisted, see data/processed/image_whitelist.json) committed to the repo
# so the deployed backend can serve them without shipping the full raw extraction folder.
IMAGES_DIR = Path(__file__).resolve().parent / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retriever (embedding model, reranker, BM25 index)...")
    state["retriever"] = HybridRetriever(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        image_collection_name=settings.image_collection_name,
    )
    logger.info("Retriever ready: %d text chunks indexed for BM25", state["retriever"].text_chunks_count)

    state["generator"] = MedLensGenerator(
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model,
    )
    logger.info("Generator ready (model=%s)", settings.groq_model)

    state["uploads"] = {}

    yield
    state.clear()


app = FastAPI(title="MedLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/api/health", response_model=HealthResponse)
def health():
    retriever: HybridRetriever = state["retriever"]
    return HealthResponse(
        status="ok",
        text_chunks_count=retriever.text_chunks_count,
        image_chunks_count=retriever.image_chunks_count,
    )


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for now.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is too large (max 10 MB).")

    chunks = parse_pdf_to_chunks(file_bytes)[:MAX_UPLOAD_CHUNKS]
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any text from this PDF.")

    state["uploads"][session_id] = {"filename": file.filename, "chunks": chunks}

    return UploadResponse(filename=file.filename, num_chunks=len(chunks))


@app.delete("/api/upload/{session_id}")
def remove_document(session_id: str):
    state["uploads"].pop(session_id, None)
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    retriever: HybridRetriever = state["retriever"]
    generator: MedLensGenerator = state["generator"]

    upload = state["uploads"].get(request.session_id) if request.session_id else None
    uploaded_chunks = select_relevant_chunks(request.message, upload["chunks"], top_k=10) if upload else None

    def event_stream():
        try:
            for event in generator.generate_stream(
                retriever, request.message, history=request.history, uploaded_chunks=uploaded_chunks
            ):
                yield _sse_event(event)
        except Exception as exc:  # surfaced to the client as an error event
            logger.exception("Error while generating response")
            yield _sse_event({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/export-pdf")
def export_pdf(request: ExportPdfRequest):
    pdf_bytes = build_pdf(request.title, request.text)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="medlens.pdf"'},
    )
