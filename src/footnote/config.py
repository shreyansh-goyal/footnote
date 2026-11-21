"""All environment-driven settings live here. Nothing else reads os.environ."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# Required: folder that contains the note subfolders to index.
_notes_dir = os.environ.get("FOOTNOTE_NOTES_DIR")
if not _notes_dir:
    raise SystemExit("Set FOOTNOTE_NOTES_DIR in .env (see .env.example).")
NOTES_DIR = Path(_notes_dir)

# NVIDIA retires models often; a 410 Gone error means pick another one.
CHAT_MODEL = os.environ.get("FOOTNOTE_CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

# QDRANT_URL set -> talk to a Qdrant server (the Docker setup). Unset -> embedded local mode.
QDRANT_URL = os.environ.get("QDRANT_URL")
DB_PATH = os.environ.get("FOOTNOTE_DB_PATH", "data/qdrant")


def get_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=DB_PATH)
