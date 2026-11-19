import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from footnote.loader import FOLDERS, NOTES_DIR, load_note
from footnote.schema import Document

MAX_CHARS = 1000
CHUNK_OVERLAP = 150
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=CHUNK_OVERLAP)


class Chunk(BaseModel):
    text: str
    title: str
    folder: str
    source: str
    heading: str


def split_by_heading(text: str, default_heading: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Lines inside ``` fences are never headings."""
    sections: list[tuple[str, list[str]]] = [(default_heading, [])]
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "".join(body).strip()]


def chunk_document(doc: Document) -> list[Chunk]:
    chunks = []
    for heading, body in split_by_heading(doc.text, default_heading=doc.title):
        for piece in SPLITTER.split_text(body):
            chunks.append(
                Chunk(
                    text=piece,
                    title=doc.title,
                    folder=doc.folder,
                    source=doc.source,
                    heading=heading,
                )
            )
    return chunks


def main() -> None:
    chunks = []
    for folder in FOLDERS:
        for path in sorted((NOTES_DIR / folder).glob("*.md")):
            chunks += chunk_document(load_note(path))
    sizes = [len(c.text) for c in chunks]
    print(f"{len(chunks)} chunks, avg {sum(sizes) // len(sizes)} chars, max {max(sizes)}")
    print(f"{sum(s > MAX_CHARS for s in sizes)} chunks are over {MAX_CHARS} chars")


if __name__ == "__main__":
    main()
