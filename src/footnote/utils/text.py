def norm(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks and capitalisation never cause a mismatch."""
    return " ".join(text.lower().split())
