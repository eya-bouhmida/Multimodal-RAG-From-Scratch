import pymupdf as fitz
import tiktoken


def extract_pdf_pages(file_bytes: bytes) -> list[dict]:
    """Extract text per page from a PDF file's raw bytes."""
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            pages.append({"page_num": page_num, "text": page.get_text()})
    return pages


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of max_tokens tokens."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max_tokens - overlap

    return chunks


def parse_pdf_to_chunks(file_bytes: bytes, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """Extract and chunk all text from a PDF, dropping empty pages."""
    chunks = []
    for page in extract_pdf_pages(file_bytes):
        if not page["text"].strip():
            continue
        chunks.extend(chunk_text(page["text"], max_tokens=max_tokens, overlap=overlap))
    return chunks
