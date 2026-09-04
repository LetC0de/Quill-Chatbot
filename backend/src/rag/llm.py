from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.prompt import title_prompt
from src.utils.settings import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY
)


async def generate_title(question: str) -> str | None:
    """Ask the model for a short ChatGPT-style conversation title.

    Cheap, non-critical, and best-effort: any failure (empty text, API error)
    returns None so callers keep their existing "New Chat" fallback rather than
    failing the whole request. Runs after the answer streams so it never delays
    the user's response.
    """
    try:
        chain = title_prompt | llm
        resp = await chain.ainvoke({"question": question})
        title = (resp.content or "").strip().strip('"').strip("'").strip()
        # Collapse whitespace; enforce a sane length so a runaway model can't
        # blow the DB column or the sidebar layout.
        title = " ".join(title.split())
        if not title or len(title) > 80:
            return None
        return title
    except Exception:
        return None
