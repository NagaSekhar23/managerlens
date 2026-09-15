"""Splits a document into retrieval-sized chunks.

Pure function, no I/O — greedily accumulates whole paragraphs (split on blank
lines) until adding the next one would exceed `max_chars`, then starts a new
chunk. Keeping paragraphs intact (rather than cutting mid-sentence) keeps
each chunk coherent enough to be useful evidence on its own.
"""


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
