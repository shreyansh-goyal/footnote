from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

MODEL = "nvidia/nemotron-3-embed-1b"

SENTENCE = "chunk_overlap makes neighbouring chunks share some characters."
QUESTION = "Why do chunks overlap?"
UNRELATED = "How do I autoscale pods on Kubernetes?"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def main() -> None:
    embedder = NVIDIAEmbeddings(model=MODEL)

    passage_vec = embedder.embed_documents([SENTENCE])[0]
    question_vec = embedder.embed_query(QUESTION)
    unrelated_vec = embedder.embed_query(UNRELATED)

    print(f"vector length: {len(passage_vec)}")
    print(f"first 5 numbers: {passage_vec[:5]}")
    print(f"question vs passage:  {cosine(question_vec, passage_vec):.3f}")
    print(f"unrelated vs passage: {cosine(unrelated_vec, passage_vec):.3f}")


if __name__ == "__main__":
    main()
