from src.pdf_export import build_pdf


def test_build_pdf_returns_valid_pdf_bytes():
    pdf_bytes = build_pdf("Test Title", "Line one.\nLine two with **bold** text.")
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100


def test_build_pdf_escapes_special_characters_without_crashing():
    pdf_bytes = build_pdf("Title", "Less than < and ampersand & sign")
    assert pdf_bytes.startswith(b"%PDF")
