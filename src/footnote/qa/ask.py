from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from footnote.qa.answer import CHAT_MODEL, answer
from footnote.core.config import get_client
from footnote.ingestion.indexer import COLLECTION, MODEL
from footnote.retrieval.retriever import retrieve


def main() -> None:
    client = get_client()
    if not client.collection_exists(COLLECTION):
        raise SystemExit("Nothing indexed yet. Run: python -m footnote.ingestion.indexer")
    embedder = NVIDIAEmbeddings(model=MODEL)
    llm = ChatNVIDIA(model=CHAT_MODEL, max_completion_tokens=4096)  # thinking tokens count against this

    while True:
        question = input("\nAsk a question (empty line to quit): ").strip()
        if not question:
            break
        print(f"\n{answer(question, retrieve(client, embedder, question), llm)}")


if __name__ == "__main__":
    main()
