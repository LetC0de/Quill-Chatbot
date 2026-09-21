"""
Generator module for evaluation - isolated generator function.

This provides a simple generate(query, context) function that uses the EXACT
same LLM and prompt as the main project (src/graph/streaming.py), for
component-level evaluation. No history, no streaming, no checkpointing.
"""

from src.rag.llm import llm
from src.rag.prompt import prompt, concierge_prompt


def generate(query: str, context: list[str]) -> str:
    """
    Generate an answer given a query and context chunks.

    Matches production exactly:
    - Uses `prompt.invoke({"context": context_str, "question": query})`
    - Uses `llm.invoke()` (sync, not stream)
    - No conversation history (eval is single-turn)

    Args:
        query: The user's question
        context: List of context chunk strings (from ideal_context in golden set)

    Returns:
        Generated answer string
    """
    if not context:
        # No context - use concierge prompt (same as production)
        final_prompt = concierge_prompt.invoke({"question": query})
    else:
        # Has context - join chunks with page labels like production _build_context
        # ideal_context doesn't have page numbers, so add sequential [Page N]
        context_parts = []
        for i, chunk in enumerate(context, 1):
            context_parts.append(f"[Page {i}] {chunk}")
        context_str = "\n\n".join(context_parts)

        # Exact same prompt invocation as streaming.py line 190
        final_prompt = prompt.invoke({"context": context_str, "question": query})

    # Use synchronous invoke (production uses astream for streaming)
    result = llm.invoke(final_prompt)
    return result.content if isinstance(result.content, str) else str(result.content)