from langchain_nvidia_ai_endpoints import ChatNVIDIA

from footnote.core.config import CHAT_MODEL  # noqa: F401  (re-exported for ask.py)

MIN_SCORE = 0.3  # below this, a hit is treated as not relevant; tune it once you see real scores

NO_ANSWER = "I don't know: your notes don't cover that."


def relevant(hits: list[dict]) -> list[dict]:
    """Drop weak hits. The order is kept, so [n] in the prompt is position n-1 in this list."""
    return [h for h in hits if h["score"] >= MIN_SCORE]


def format_sources(hits: list[dict]) -> tuple[str, str]:
    """Number the hits once and return (prompt_block, footnotes) so [n] means the same in both."""
    numbered = list(enumerate(hits, start=1))
    block = "\n\n".join(f"[{i}] {h['title']} > {h['heading']}\n{h['text']}" for i, h in numbered)
    footnotes = "\n".join(f"  [{i}] {h['source']} > {h['heading']}" for i, h in numbered)
    return block, footnotes


def build_prompt(question: str, sources: str) -> str:
    """Wrap the numbered source block and the question in instructions that force [n] citations."""
    return (
        "Answer the question using ONLY the numbered sources below.\n"
        "Cite the sources you used inline like [1] or [2][3].\n"
        f"If the sources don't contain the answer, reply exactly: {NO_ANSWER}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {question}\nAnswer:"
    )


def answer(question: str, hits: list[dict], llm: ChatNVIDIA) -> str:
    hits = relevant(hits)
    if not hits:
        return NO_ANSWER  # nothing relevant retrieved: don't ask the model to guess
    block, footnotes = format_sources(hits)  # footnotes come from our list, not the model
    reply = llm.invoke(build_prompt(question, block))
    if not reply.content.strip():
        # Reasoning models can spend the whole token budget thinking and never write the answer.
        reason = reply.response_metadata.get("finish_reason")
        return f"The model returned no answer (finish_reason={reason!r}). Try again or raise max_completion_tokens."
    return f"{reply.content}\n\nSources:\n{footnotes}"
