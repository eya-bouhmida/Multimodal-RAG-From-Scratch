import pymupdf

from src.parser import chunk_text, extract_pdf_pages, parse_pdf_to_chunks


def _make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox((72, 72, 500, 700), text, fontsize=11)
    return doc.tobytes()


def test_chunk_text_splits_long_text_with_overlap():
    text = "word " * 1000
    chunks = chunk_text(text, max_tokens=50, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_returns_single_chunk_for_short_text():
    chunks = chunk_text("hello world", max_tokens=500, overlap=50)
    assert len(chunks) == 1


def test_extract_pdf_pages_returns_text_per_page():
    pdf_bytes = _make_pdf_bytes("Hello MedLens test page.")
    pages = extract_pdf_pages(pdf_bytes)
    assert len(pages) == 1
    assert pages[0]["page_num"] == 1
    assert "Hello MedLens" in pages[0]["text"]


def test_parse_pdf_to_chunks_skips_empty_pages():
    doc = pymupdf.open()
    doc.new_page()  # empty page, should be skipped
    page = doc.new_page()
    page.insert_textbox((72, 72, 500, 700), "Some real content here.", fontsize=11)
    pdf_bytes = doc.tobytes()

    chunks = parse_pdf_to_chunks(pdf_bytes)
    assert len(chunks) == 1
    assert "Some real content" in chunks[0]
