from datetime import date
from typing_extensions import Literal

from pydantic import BaseModel


class Document(BaseModel):
    title: str
    text: str
    folder: str 
    source: str
    source_type: Literal["obsidian", "medium"] = "obsidian"