import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_schema_embeddings(tables: dict) -> dict:
    model = _get_model()
    embeddings = {}
    for table_name, meta in tables.items():
        cols = ", ".join(
            f"{c['name']} ({c['type']})" for c in meta["columns"]
        )
        doc = (
            f"Table: {table_name}. "
            f"Columns: {cols}. "
            f"Sample data: {meta['sample_rows'][:300]}"
        )
        embeddings[table_name] = {
            "embedding": model.encode(doc, normalize_embeddings=True),
            "doc": doc,
        }
    return embeddings


def retrieve_relevant_tables(
    question: str,
    schema_embeddings: dict,
    tables: dict,
    top_k: int = 3,
    threshold: float = 0.15,
) -> dict:
    if not schema_embeddings:
        return tables

    # Always return all if small enough
    if len(tables) <= 4:
        return tables

    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    scores = {}
    for table_name, meta in schema_embeddings.items():
        score = float(np.dot(q_emb, meta["embedding"]))
        scores[table_name] = score

    sorted_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, s in sorted_tables[:top_k] if s >= threshold]

    # Fallback: if nothing passes threshold, return all
    if not selected:
        return tables

    return {t: tables[t] for t in selected if t in tables}