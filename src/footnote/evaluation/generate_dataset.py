"""Build a synthetic evaluation set from your own chunks.

Pipeline:   pick chunks -> LLM writes a question + answer -> code checks -> LLM judge scores -> keep

  single-note: one chunk  -> one question it answers
  multi-note:  a chunk plus its nearest neighbour from a DIFFERENT note -> one question
               that needs both (the neighbour is found with the vectors already in Qdrant)

The label is the chunk (or chunks) the question was written from, so it is known by
construction. Output: data/golden_synthetic.jsonl (kept) and data/golden_rejected.jsonl.
"""

import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from footnote.ingestion.chunker import Chunk, chunk_document
from footnote.core.config import CHAT_MODEL, GEN_MODEL, get_client
from footnote.ingestion.indexer import COLLECTION, chunk_id, load_notes
from footnote.utils.text import norm

OUT_PATH = Path("data/golden_synthetic.jsonl")
REJECT_PATH = Path("data/golden_rejected.jsonl")
N_SINGLE = 20
N_MULTI = 10
MIN_CHARS = 250  # skip tiny chunks: too little to ask a real question about
MIN_JUDGE_SCORE = 4  # judge scores 0-5; anything below this is dropped
COPY_WORDS = 6  # a question sharing this many words in a row with a chunk is copying it
WORKERS = 4
SEED = 7

# Variety, so the questions look like different people asking in different ways.
PERSONAS = [
    "a learner revising a topic before an interview",
    "an engineer who wants a quick reminder while building something",
    "a newcomer who knows little and asks basic questions",
]
STYLES = ["clear and grammatical", "casual and a little sloppy", "terse, like a search-engine query"]
LENGTHS = ["short (under 12 words)", "medium (12 to 25 words)", "long (25 to 50 words)"]

# Generic examples (not from your vault). Swap in real handwritten ones, but never use
# held-out questions here, or the held-out set stops being independent.
FEW_SHOT = """Example (one excerpt):
Excerpt: "A Horizontal Pod Autoscaler compares observed CPU use with the target and scales the replica count proportionally, checking every 15 seconds by default."
Output: {"question": "how often does kubernetes recheck whether to change pod count", "answer": "Every 15 seconds by default."}

Example (two excerpts):
[1] "Overlap repeats the end of one chunk at the start of the next, so a sentence cut at the boundary stays whole in one chunk."
[2] "Smaller chunks make retrieval more precise but give the model less context per hit."
Output: {"question": "If I shrink my chunks, how does overlap still protect against split sentences, and what do I lose in context?", "answer": "Overlap keeps a boundary-cut sentence intact in one chunk, while smaller chunks give the model less context per hit."}"""

GEN_RULES = """- Paraphrase. Do not reuse the excerpt wording in the question.
- The question must stand alone: never say "the excerpt", "this note", "the text" or "the passage".
- The answer must use only the excerpt(s).
- Reply with one JSON object and nothing else: {{"question": ..., "answer": ...}}"""

SINGLE_PROMPT = """You write test questions for a search system over someone's personal notes.
The asker is {persona}. Write the question in a {style} style, {length}.
Given ONE note excerpt, write ONE question that it answers, plus the answer.
Rules:
""" + GEN_RULES + """

{few_shot}

Now the real excerpt (from {title} > {heading}):
\"\"\"{text}\"\"\"
Output:"""

MULTI_PROMPT = """You write test questions for a search system over someone's personal notes.
The asker is {persona}. Write the question in a {style} style, {length}.
Given TWO note excerpts, write ONE question that can only be answered by combining facts from BOTH, plus the answer.
Rules:
""" + GEN_RULES + """

{few_shot}

Now the real excerpts:
[1] (from {title_a} > {heading_a})
\"\"\"{text_a}\"\"\"
[2] (from {title_b} > {heading_b})
\"\"\"{text_b}\"\"\"
Output:"""

JUDGE_PROMPT = """You grade test questions for a search system over someone's personal notes.
Score 0 to 5:
  5 = realistic, stands alone, and fully answerable from the excerpt(s){multi}
  3 = usable but vague, or slightly off
  0 = nonsense, or not answerable from the excerpt(s)
{multi_rule}
Question: {question}
Proposed answer: {answer}

Excerpts:
{excerpts}

Reply with one JSON object and nothing else: {{"score": <0-5>, "reason": "<one short sentence>"}}"""

SELF_REFERENCE = re.compile(r"\b(the|this) (excerpt|note|text|passage|section)\b", re.I)


@dataclass(frozen=True)
class Task:
    kind: str  # "single" or "multi"
    chunks: tuple[Chunk, ...]
    persona: str
    style: str
    length: str


def parse(reply: str) -> dict | None:
    """Pull the first JSON object out of a model reply (tolerates ``` fences and chatter)."""
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def copies_text(question: str, texts: list[str], n: int = COPY_WORDS) -> bool:
    """True if the question repeats n or more consecutive words from any of the chunk texts."""
    words = norm(question).split()
    haystacks = [norm(t) for t in texts]
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        if any(gram in h for h in haystacks):
            return True
    return False


def rule_reject(data: dict | None, texts: list[str]) -> str | None:
    """Cheap code checks on a generated case. Returns the reason it is unusable, or None."""
    if not data or not all(isinstance(data.get(k), str) and data[k].strip() for k in ("question", "answer")):
        return "bad json"
    if SELF_REFERENCE.search(data["question"]):
        return "question not standalone"
    if copies_text(data["question"], texts):
        return "question copies the chunk wording"
    return None


def in_index(client: QdrantClient, chunk: Chunk) -> bool:
    """Label check: this exact chunk (same source + text) is really stored in the index."""
    return bool(client.retrieve(COLLECTION, ids=[chunk_id(chunk)]))


def neighbour(client: QdrantClient, chunk: Chunk) -> Chunk | None:
    """Closest chunk from a DIFFERENT note, using the vector already stored (no embedding call)."""
    stored = client.retrieve(COLLECTION, ids=[chunk_id(chunk)], with_vectors=True)
    if not stored:
        return None
    other_notes = Filter(must_not=[FieldCondition(key="source", match=MatchValue(value=chunk.source))])
    points = client.query_points(COLLECTION, query=stored[0].vector, limit=1, query_filter=other_notes).points
    if not points:
        return None
    payload = points[0].payload
    return Chunk(**{field: payload[field] for field in Chunk.model_fields})


def build_tasks(client: QdrantClient, chunks: list[Chunk], n_single: int, n_multi: int, rng: random.Random):
    """Pick anchor chunks (already shuffled) and pair multi-note ones with a neighbour."""
    tasks, skipped = [], Counter()
    anchors = iter(chunks)

    def style() -> tuple[str, str, str]:
        return rng.choice(PERSONAS), rng.choice(STYLES), rng.choice(LENGTHS)

    for want, kind in ((n_single, "single"), (n_multi, "multi")):
        made = 0
        for chunk in anchors:
            if made == want:
                break
            if not in_index(client, chunk):
                skipped["chunk not in index"] += 1  # notes changed since the last indexing run
                continue
            if kind == "single":
                tasks.append(Task("single", (chunk,), *style()))
            else:
                other = neighbour(client, chunk)
                if other is None:
                    skipped["no neighbour from another note"] += 1
                    continue
                tasks.append(Task("multi", (chunk, other), *style()))
            made += 1
    return tasks, skipped


def write_case(llm: ChatNVIDIA, task: Task) -> dict | None:
    if task.kind == "single":
        c = task.chunks[0]
        prompt = SINGLE_PROMPT.format(
            persona=task.persona, style=task.style, length=task.length, few_shot=FEW_SHOT,
            title=c.title, heading=c.heading, text=c.text,
        )
    else:
        a, b = task.chunks
        prompt = MULTI_PROMPT.format(
            persona=task.persona, style=task.style, length=task.length, few_shot=FEW_SHOT,
            title_a=a.title, heading_a=a.heading, text_a=a.text,
            title_b=b.title, heading_b=b.heading, text_b=b.text,
        )
    return parse(llm.invoke(prompt).content)


def judge(llm: ChatNVIDIA, task: Task, data: dict) -> tuple[int | None, str]:
    multi = task.kind == "multi"
    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(task.chunks, start=1))
    prompt = JUDGE_PROMPT.format(
        multi=" and genuinely needs BOTH excerpts" if multi else "",
        multi_rule="If one excerpt alone is enough to answer, the score is at most 2.\n" if multi else "",
        question=data["question"], answer=data["answer"], excerpts=excerpts,
    )
    verdict = parse(llm.invoke(prompt).content)
    if not verdict or not isinstance(verdict.get("score"), (int, float)):
        return None, "judge returned bad json"
    return max(0, min(5, int(verdict["score"]))), str(verdict.get("reason", ""))


def process(task: Task, gen_llm: ChatNVIDIA, judge_llm: ChatNVIDIA) -> dict:
    """Generate, check and judge one case. Always returns a record; 'rejected' says why not kept."""
    record = {
        "kind": task.kind, "persona": task.persona, "style": task.style, "length": task.length,
        "passages": [{"source": c.source, "heading": c.heading, "text": c.text} for c in task.chunks],
    }
    data = write_case(gen_llm, task)
    reason = rule_reject(data, [c.text for c in task.chunks])
    if reason:
        return record | {"question": (data or {}).get("question"), "rejected": reason}
    score, why = judge(judge_llm, task, data)
    record |= {"question": data["question"].strip(), "answer": data["answer"].strip(), "score": score, "judge_reason": why}
    if score is None or score < MIN_JUDGE_SCORE:
        record["rejected"] = f"judge score {score}"
    return record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def generate(n_single: int = N_SINGLE, n_multi: int = N_MULTI, out_path: Path = OUT_PATH, reject_path: Path = REJECT_PATH) -> None:
    client = get_client()
    if not client.collection_exists(COLLECTION):
        raise SystemExit("Nothing indexed yet. Run: python -m footnote.ingestion.indexer")
    chunks = [c for doc in load_notes().values() for c in chunk_document(doc) if len(c.text) >= MIN_CHARS]
    rng = random.Random(SEED)
    rng.shuffle(chunks)

    tasks, skipped = build_tasks(client, chunks, n_single, n_multi, rng)
    # Writer and judge are different models so the judge isn't grading its own writing.
    # Thinking off: writing a question or giving a score doesn't need it, and it is slower and costlier.
    no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}
    gen_llm = ChatNVIDIA(model=GEN_MODEL, max_completion_tokens=2048, model_kwargs=no_thinking)
    judge_llm = ChatNVIDIA(model=CHAT_MODEL, max_completion_tokens=2048, model_kwargs=no_thinking)

    kept, rejected = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process, t, gen_llm, judge_llm) for t in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            (rejected if "rejected" in record else kept).append(record)
            print(f"[{done}/{len(tasks)}] kept={len(kept)} rejected={len(rejected)}")

    write_jsonl(out_path, kept)
    write_jsonl(reject_path, rejected)
    reasons = Counter(r["rejected"] for r in rejected) + skipped
    print(f"\nwrote {len(kept)} cases to {out_path} ({sum(k['kind'] == 'multi' for k in kept)} multi-note)")
    print(f"rejected/skipped: {dict(reasons) or 'none'}  (details in {reject_path})")


if __name__ == "__main__":
    generate()
