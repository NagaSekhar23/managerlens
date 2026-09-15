-- Runs automatically on first container startup (docker-entrypoint-initdb.d convention).
-- The knowledge_chunks table itself is created by the app (scripts/ingest_knowledge.py
-- calls Base.metadata.create_all), not here — this only enables the extension it needs.
CREATE EXTENSION IF NOT EXISTS vector;
