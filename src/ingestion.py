import re
import sqlite3
import pandas as pd
import numpy as np
from typing import Any

# Lazy-loaded to avoid importing sentence_transformers at module level
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def sanitize_table_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "table"


def infer_relationships(tables: dict) -> list[dict]:
    """
    Detect likely JOIN keys across tables using a two-signal approach:
      1. Semantic embedding similarity of column names (catches empid ↔ employee_id)
      2. Token overlap score (catches partial name matches)
    Both signals are combined; a pair scores above threshold → flagged as relationship.
    """
    model = _get_embed_model()
    table_names = list(tables.keys())
    relationships = []

    # Pre-embed all column names across all tables
    col_embeddings: dict[str, dict[str, Any]] = {}
    for tname in table_names:
        for col in tables[tname]["columns"]:
            cname = col["name"]
            key = f"{tname}.{cname}"
            col_embeddings[key] = {
                "table": tname,
                "col": cname,
                "type": col["type"],
                "embedding": model.encode(
                    _col_label(tname, cname), normalize_embeddings=True
                ),
            }

    # Compare every cross-table column pair
    keys = list(col_embeddings.keys())
    for i, k1 in enumerate(keys):
        m1 = col_embeddings[k1]
        for k2 in keys[i + 1:]:
            m2 = col_embeddings[k2]
            if m1["table"] == m2["table"]:
                continue
            if not _types_compatible(m1["type"], m2["type"]):
                continue

            sem_score = float(np.dot(m1["embedding"], m2["embedding"]))
            tok_score = _token_overlap(m1["col"], m2["col"])
            combined = 0.6 * sem_score + 0.4 * tok_score

            if combined >= 0.55:
                relationships.append({
                    "table_a": m1["table"],
                    "col_a": m1["col"],
                    "table_b": m2["table"],
                    "col_b": m2["col"],
                    "score": round(combined, 3),
                })

    # Deduplicate: keep highest-scoring pair per table combination
    seen: dict[tuple, dict] = {}
    for rel in sorted(relationships, key=lambda r: r["score"], reverse=True):
        pair = tuple(sorted([rel["table_a"], rel["table_b"]]))
        col_pair = (rel["col_a"], rel["col_b"])
        dedup_key = (pair, col_pair)
        if dedup_key not in seen:
            seen[dedup_key] = rel

    return list(seen.values())


def _col_label(table: str, col: str) -> str:
    """
    Create a human-readable label for embedding.
    e.g. table=employee_data, col=empid → 'employee id identifier'
    """
    combined = f"{table} {col}".replace("_", " ")
    return combined


def _token_overlap(c1: str, c2: str) -> float:
    """Jaccard-style token overlap between two column names."""
    def tokenize(s):
        s = re.sub(r"[^a-z0-9]", " ", s.lower())
        tokens = set(s.split())
        # Also add substrings of length 3+ to catch 'emp' in 'employee'
        for t in list(tokens):
            if len(t) >= 4:
                for size in range(3, len(t)):
                    tokens.add(t[:size])
        return tokens

    t1, t2 = tokenize(c1), tokenize(c2)
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union)


def _types_compatible(t1: str, t2: str) -> bool:
    numeric = {"integer", "real", "numeric", "float", "int", "bigint", "double"}
    text = {"text", "varchar", "char", "string", "nvarchar"}
    t1, t2 = t1.lower(), t2.lower()
    if t1 == t2:
        return True
    if t1 in numeric and t2 in numeric:
        return True
    if t1 in text and t2 in text:
        return True
    # Allow TEXT ↔ INTEGER for id columns (common in practice)
    return False

def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    DATE_FORMATS = [
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%B %d, %Y",
    ]

    for col in df.columns:
        # catch both plain object AND pandas StringDtype
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue

        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue

        matched_fmt = None
        for fmt in DATE_FORMATS:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors="raise")
                if parsed.notna().mean() >= 0.8:
                    matched_fmt = fmt
                    break
            except Exception:
                continue

        if matched_fmt:
            df[col] = pd.to_datetime(
                df[col], format=matched_fmt, errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    return df

def load_csvs(uploaded_files: list, conn: sqlite3.Connection) -> dict[str, Any]:
    tables = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            df = normalize_date_columns(df)
            df.columns = [
                re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in df.columns
            ]
            # Collapse multiple underscores
            df.columns = [re.sub(r"_+", "_", c).strip("_") for c in df.columns]

            table_name = sanitize_table_name(f.name)
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [
                {"name": row[1], "type": row[2] or "TEXT"} for row in cursor.fetchall()
            ]

            sample_df = df.head(3)
            sample_rows = sample_df.to_string(index=False)

            tables[table_name] = {
                "columns": columns,
                "sample_rows": sample_rows,
                "row_count": len(df),
                "original_filename": f.name,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load {f.name}: {e}")

    return tables