"""Ingests backend/data/knowledge/*.md into the knowledge_chunks table.

Usage (from backend/):
    ../.venv/bin/python -m scripts.ingest_knowledge

Idempotent: re-running deletes and re-inserts chunks for each source file it
processes, so editing a knowledge doc and re-running keeps the table in sync
without manual cleanup or duplicate rows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Base, KnowledgeChunk  # noqa: E402
from app.db.session import engine, SessionLocal  # noqa: E402
from app.services.chunking import chunk_text  # noqa: E402
from app.services.gemini_client import embed_text  # noqa: E402

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def ingest_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    title = _extract_title(text, fallback=path.stem.replace("_", " ").title())
    chunks = chunk_text(text)

    with SessionLocal() as session:
        session.query(KnowledgeChunk).filter(KnowledgeChunk.source_file == path.name).delete()

        for index, chunk in enumerate(chunks):
            embedding = embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
            session.add(
                KnowledgeChunk(
                    source_file=path.name,
                    source_title=title,
                    chunk_index=index,
                    content=chunk,
                    embedding=embedding,
                )
            )

        session.commit()

    return len(chunks)


def main() -> None:
    Base.metadata.create_all(engine)

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        print(f"No knowledge files found in {KNOWLEDGE_DIR}")
        return

    total = 0
    for path in files:
        count = ingest_file(path)
        total += count
        print(f"  {path.name}: {count} chunks")

    print(f"Ingested {total} chunks from {len(files)} files.")


if __name__ == "__main__":
    main()
