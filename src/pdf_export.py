import re
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _to_markup(line: str) -> str:
    """Escape XML entities, then convert simple **bold** markdown to reportlab's mini-markup."""
    escaped = escape(line)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def build_pdf(title: str, body: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = [Paragraph(_to_markup(title), styles["Title"]), Spacer(1, 0.5 * cm)]

    for paragraph in body.split("\n"):
        if paragraph.strip():
            story.append(Paragraph(_to_markup(paragraph), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    return buffer.getvalue()
