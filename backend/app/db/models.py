from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.services.embedding_config import EMBEDDING_DIMENSIONS


class Base(DeclarativeBase):
    pass


class KnowledgeChunk(Base):
    """One chunk of a curated management-guidance document, with its embedding.

    This is our RAG knowledge base: general management guidance, not a record
    of any specific employee, team, or company policy.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_file: Mapped[str] = mapped_column(index=True)
    source_title: Mapped[str]
    chunk_index: Mapped[int]
    content: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
