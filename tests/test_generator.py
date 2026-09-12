from src.generator import MedLensGenerator, _cap_chunks_to_tokens, is_emergency_query


def test_is_emergency_query_detects_french_and_english():
    assert is_emergency_query("j'ai avalé du javel par erreur")
    assert is_emergency_query("I think I'm having a heart attack")


def test_is_emergency_query_ignores_normal_questions():
    assert not is_emergency_query("what are the symptoms of diabetes?")
    assert not is_emergency_query("quels sont les traitements pour l'hypertension?")


def test_cap_chunks_to_tokens_stops_at_budget():
    chunks = ["word " * 100 for _ in range(10)]  # ~100 tokens each
    capped = _cap_chunks_to_tokens(chunks, max_tokens=250)
    assert 0 < len(capped) < len(chunks)


def test_cap_chunks_to_tokens_keeps_everything_under_budget():
    chunks = ["short chunk"] * 3
    capped = _cap_chunks_to_tokens(chunks, max_tokens=1000)
    assert capped == chunks


def test_build_prompt_includes_uploaded_document_section():
    generator = MedLensGenerator(groq_api_key="test-key")
    prompt, sources = generator.build_prompt(
        query="Summarize this",
        text_chunks=[],
        image_results=[],
        uploaded_chunks=["This is the uploaded content."],
    )
    assert "UPLOADED DOCUMENT" in prompt
    assert "This is the uploaded content." in prompt
    assert sources == []


def test_build_prompt_never_asks_for_source_citations():
    generator = MedLensGenerator(groq_api_key="test-key")
    prompt, _ = generator.build_prompt(query="test", text_chunks=[], image_results=[])
    assert "cite the source" not in prompt.lower()
