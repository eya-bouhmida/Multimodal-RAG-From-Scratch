import tiktoken
from groq import Groq

_ENC = tiktoken.get_encoding("cl100k_base")

# Groq free tier caps this model at 8000 tokens/minute (input + output combined),
# so uploaded-document content must be capped hard to avoid 413 rate-limit errors.
UPLOADED_DOC_TOKEN_BUDGET = 2500
UPLOADED_DOC_TOP_K = 3
UPLOADED_DOC_TOP_K_IMAGES = 1
UPLOADED_DOC_MAX_OUTPUT_TOKENS = 700
DEFAULT_MAX_OUTPUT_TOKENS = 1200


def _cap_chunks_to_tokens(chunks: list[str], max_tokens: int) -> list[str]:
    capped = []
    used = 0
    for chunk in chunks:
        chunk_tokens = len(_ENC.encode(chunk))
        if used + chunk_tokens > max_tokens:
            break
        capped.append(chunk)
        used += chunk_tokens
    return capped


EMERGENCY_KEYWORDS = [
    # French
    "urgence", "j'ai avalé", "j'ai bu du", "avalé du", "intoxication", "empoisonn",
    "brûlure grave", "hémorragie", "saigne abondam", "ne respire plus", "inconscient",
    "arrêt cardiaque", "overdose", "surdose", "envie de mourir", "je veux mourir",
    "je veux me suicider", "étouffe", "convulsion", "javel", "poison", "crise cardiaque",
    "douleur thoracique", "paralysie soudaine",
    # English
    "emergency", "swallowed", "poisoning", "poisoned", "overdose", "can't breathe",
    "cannot breathe", "unconscious", "severe bleeding", "heart attack", "stroke",
    "suicidal", "want to kill myself", "want to die", "choking", "seizure", "bleach",
    "chest pain",
]

EMERGENCY_MESSAGE = (
    "🚨 If this may be a medical emergency, call emergency services immediately "
    "(190 SAMU in Tunisia, 15 or 112 in France, 911 in the US) or go to the nearest "
    "emergency room — do not wait for a chatbot response.\n\n"
    "🚨 S'il s'agit potentiellement d'une urgence médicale, appelle immédiatement les "
    "urgences (190 SAMU en Tunisie, 15 ou 112 en France) ou rends-toi aux urgences les "
    "plus proches — n'attends pas la réponse d'un chatbot.\n\n"
)


def is_emergency_query(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


SYSTEM_RULES = """You are MedLens, an expert bilingual medical assistant (French/English).

ABSOLUTE RULES:
1. Answer ONLY using information from the documents provided below.
2. Do NOT use your own general knowledge under any circumstances.
3. If the information is not in the documents, say exactly: "I cannot find this information in the available documents."
4. Do NOT mention filenames, page numbers, or any source references in your answer. Write a clean, natural response as if you simply know the answer.
5. Always respond in the same language as the question.
6. Be precise, clear, and helpful — you are helping non-medical people understand health information."""


class MedLensGenerator:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def build_prompt(
        self, query: str, text_chunks, image_results, history=None, uploaded_chunks=None
    ) -> tuple[str, list[str]]:
        context = "=== MEDICAL DOCUMENTS ===\n"
        sources = []

        for i, (score, result) in enumerate(text_chunks):
            payload = result[1]["payload"]
            context += f"\n--- Document {i + 1} ---\n"
            context += f"Source: {payload['filename']} (Page {payload['page_num']})\n"
            context += f"{payload['text']}\n"
            sources.append(f"{payload['filename']} p.{payload['page_num']}")

        if image_results:
            context += "\n=== MEDICAL FIGURES AND IMAGES ===\n"
            for i, img in enumerate(image_results):
                context += f"\n--- Figure {i + 1} ---\n"
                context += f"Image: {img.payload['filename']}\n"
                context += f"Description: {img.payload['text']}\n"
                sources.append(f"Figure: {img.payload['filename']}")

        if uploaded_chunks:
            context += "\n=== UPLOADED DOCUMENT (prioritize this if relevant to the question) ===\n"
            for i, chunk in enumerate(uploaded_chunks):
                context += f"\n--- Excerpt {i + 1} ---\n{chunk}\n"

        history_block = ""
        if history:
            recent = history[-6:]
            lines = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
            history_block = f"\nPREVIOUS CONVERSATION (for context only, still answer only from the documents above):\n{lines}\n"

        prompt = f"""{SYSTEM_RULES}

MEDICAL CONTEXT:
{context}
{history_block}
QUESTION: {query}

CLEAR RESPONSE (no source citations):"""

        return prompt, sources

    def generate_stream(
        self, retriever, query: str, history=None, top_k: int = 8, top_k_images: int = 3, uploaded_chunks=None
    ):
        if is_emergency_query(query):
            yield {"type": "token", "content": EMERGENCY_MESSAGE}

        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        if uploaded_chunks:
            top_k = min(top_k, UPLOADED_DOC_TOP_K)
            top_k_images = min(top_k_images, UPLOADED_DOC_TOP_K_IMAGES)
            uploaded_chunks = _cap_chunks_to_tokens(uploaded_chunks, UPLOADED_DOC_TOKEN_BUDGET)
            max_output_tokens = UPLOADED_DOC_MAX_OUTPUT_TOKENS

        text_chunks = retriever.hybrid_search(query, top_k=top_k)
        image_results = retriever.image_search(query, top_k=top_k_images)
        prompt, sources = self.build_prompt(
            query, text_chunks, image_results, history=history, uploaded_chunks=uploaded_chunks
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=0.05,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"type": "token", "content": delta}

        yield {
            "type": "done",
            "sources": sources,
            "text_chunks": len(text_chunks),
            "images_used": len(image_results),
            "images": [
                {"filename": img.payload["filename"], "caption": img.payload["text"]} for img in image_results
            ],
        }
