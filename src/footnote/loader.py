import re
from pathlib import Path
from footnote.config import NOTES_DIR
from footnote.schema import Document

FOLDERS = ["RAG Development", "RAG Evaluation", "RAG on Kubernetes"]


def clean_wikilinks(text: str) -> str:
    # Strip the [[ ]] brackets but keep everything inside, e.g. [[link|display]] -> link|display
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def load_note(path: Path) -> Document:
    text = clean_wikilinks(path.read_text(encoding="utf-8"))
    return Document(
        title=path.stem,
        text=text,
        source=path.name,
        folder=path.parent.name,
        source_type="obsidian",
    )



def main() -> None:
    for folder in FOLDERS:
        for path in sorted((NOTES_DIR / folder).glob("*.md")):
            doc = load_note(path)
            print(f"{doc.folder} | {doc.title}: {len(doc.text.split())} words")


if __name__ == "__main__":
    main()
