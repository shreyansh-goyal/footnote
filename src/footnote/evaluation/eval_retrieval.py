import json
from dataclasses import dataclass
from pathlib import Path

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from footnote.core.config import get_client
from footnote.evaluation.generate_dataset import OUT_PATH as SYNTHETIC_PATH
from footnote.ingestion.indexer import MODEL
from footnote.retrieval.retriever import retrieve
from footnote.utils.text import norm

K = 4


@dataclass(frozen=True)
class Case:
    question: str
    # Each passage is (source file, text). It counts as found when a retrieved chunk comes from
    # that file AND contains that text. Handwritten: a short phrase. Synthetic: the whole chunk.
    passages: tuple[tuple[str, str], ...]
    held_out: bool = False  # True = never tune settings against this case
    origin: str = "handwritten"  # "handwritten" or "synthetic"


def hand(question: str, source: str, phrase: str, held_out: bool = False) -> Case:
    """A handwritten single-passage case: the retrieved chunk must contain `phrase`."""
    return Case(question, ((source, phrase),), held_out)


# The trusted core, written by hand. The synthetic set (generate_dataset.py) supplies the volume.
# Phrases are copied from the notes as indexed (plain words, no formatting). Questions paraphrase
# the note instead of reusing its wording. The held_out ones are never tuned against or used as
# few-shot examples.
QUESTIONS: list[Case] = [
    # tuning
    hand("What stops a sentence from being lost when the splitter happens to cut through it?",
         "05 Documents and Chunking.md", "Like overlapping photos in a panorama"),
    hand("What happens to a chunk that is mostly a table of figures?",
         "05 Documents and Chunking.md", "drop chunks that are mostly numbers"),
    hand("Why can't I embed my questions with one model and my chunks with another?",
         "06 Embeddings.md", "Different models = different maps, not comparable"),
    hand("Would a GPT-style model be the right thing to turn my notes into search vectors?",
         "06 Embeddings.md", "embedding models encoder-only"),
    # held-out: never tune against these
    hand("What do I lose when each section is summarised separately and the results are merged?",
         "05 Documents and Chunking.md", "Chunks summarised in isolation, connections lost", held_out=True),
    hand("Why doesn't the size of a text matter when two vectors are compared by their angle?",
         "06 Embeddings.md", "same direction = similar, whatever their length", held_out=True),
    hand("How can I arrange retrieved context so the model doesn't overlook the most useful parts?",
         "07 Vector Stores and RAG.md", "of a long prompt and miss the middle", held_out=True),
    hand("If I edit a note and only add the new version, why might answers still reflect the old one?",
         "07 Vector Stores and RAG.md", "the bot answers from outdated text", held_out=True),
]


def load_synthetic(path: Path = SYNTHETIC_PATH) -> list[Case]:
    if not path.exists():
        return []
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return [
        Case(r["question"], tuple((p["source"], p["text"]) for p in r["passages"]), origin="synthetic")
        for r in rows
    ]


def found_ranks(hits: list[dict], case: Case) -> list[int | None]:
    """For each expected passage: 1-based rank of the first retrieved chunk that matches it, else None."""
    ranks = []
    for source, text in case.passages:
        needle = norm(text)
        ranks.append(
            next(
                (r for r, h in enumerate(hits, start=1) if h["source"] == source and needle in norm(h["text"] or "")),
                None,
            )
        )
    return ranks


def group_of(case: Case) -> str:
    if case.held_out:
        return "held-out"
    if case.origin == "synthetic":
        return "synthetic-multi" if len(case.passages) > 1 else "synthetic-single"
    return "tuning"


def report(name: str, results: list[list[int | None]]) -> None:
    """hit: any passage found. MRR: 1/rank of the best one. recall: share of passages found."""
    if not results:
        return
    n = len(results)
    found = [[r for r in ranks if r is not None] for ranks in results]
    hit = sum(bool(f) for f in found)
    mrr = sum(1 / min(f) for f in found if f) / n
    recall = sum(len(f) / len(ranks) for f, ranks in zip(found, results)) / n
    print(f"{name:<17} n={n:<3} hit@{K} {hit}/{n}  MRR {mrr:.2f}  recall@{K} {recall:.2f}")


def main() -> None:
    client = get_client()
    embedder = NVIDIAEmbeddings(model=MODEL)
    results: dict[str, list[list[int | None]]] = {
        g: [] for g in ("tuning", "held-out", "synthetic-single", "synthetic-multi")
    }

    for case in QUESTIONS + load_synthetic():
        hits = retrieve(client, embedder, case.question, k=K)
        ranks = found_ranks(hits, case)
        results[group_of(case)].append(ranks)
        found = sum(r is not None for r in ranks)
        status = "OK  " if found == len(ranks) else ("PART" if found else "MISS")
        print(f"{status} ranks={ranks}  {case.question}")
        if found < len(ranks):
            print(f"      wanted {[s for s, _ in case.passages]}; got: {', '.join(h['source'] for h in hits)}")

    print()
    for group, group_results in results.items():
        report(group, group_results)


if __name__ == "__main__":
    main()
