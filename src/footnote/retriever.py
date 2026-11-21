from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient

from footnote.indexer import COLLECTION


def retrieve(client: QdrantClient, embedder: NVIDIAEmbeddings, question: str, k: int = 4) -> list[dict]:
    """Return the top-k chunks for a question as plain dicts, ready to go into a prompt."""
    hits = client.query_points(COLLECTION, query=embedder.embed_query(question), limit=k).points
    return [
        {
            "score": hit.score,
            "title": hit.payload.get("title"),
            "heading": hit.payload.get("heading"),
            "source": hit.payload.get("source"),
            "text": hit.payload.get("text"),
        }
        for hit in hits
    ]
