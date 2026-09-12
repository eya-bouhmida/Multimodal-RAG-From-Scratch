from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    text_chunks_count: int
    image_chunks_count: int


class UploadResponse(BaseModel):
    filename: str
    num_chunks: int


class ExportPdfRequest(BaseModel):
    title: str
    text: str
