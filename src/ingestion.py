import re
import sqlite3
import pandas as pd
from typing import Any


def sanitize_table_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "table"


def infer_relationships(tables: dict) -> list[dict]:
    relationships = []
    table_names = list(tables.keys())
    for i, t1 in enumerate(table_names):
        cols1 = {c["name"]: c["type"] for c in tables[t1]["columns"]}
        for t2 in table_names[i + 1 :]:
            cols2 = {c["name"]: c["type"] for c in tables[t2]["columns"]}
            for col1, type1 in cols1.items():
                for col2, type2 in cols2.items():
                    if _types_compatible(type1, type2) and _names_related(col1, col2):
                        relationships.append(
                            {"table_a": t1, "col_a": col1, "table_b": t2, "col_b": col2}
                        )
    return relationships


def _types_compatible(t1: str, t2: str) -> bool:
    numeric = {"integer", "real", "numeric", "float", "int", "bigint"}
    text = {"text", "varchar", "char", "string"}
    t1, t2 = t1.lower(), t2.lower()
    if t1 == t2:
        return True
    if t1 in numeric and t2 in numeric:
        return True
    if t1 in text and t2 in text:
        return True
    return False


def _names_related(c1: str, c2: str) -> bool:
    c1, c2 = c1.lower(), c2.lower()
    if c1 == c2:
        return True
    if c1.endswith("_id") or c2.endswith("_id"):
        base1 = c1.replace("_id", "")
        base2 = c2.replace("_id", "")
        if base1 in base2 or base2 in base1:
            return True
        if c1 == "id" and c2.endswith("_id"):
            return True
        if c2 == "id" and c1.endswith("_id"):
            return True
    return False


def load_csvs(uploaded_files: list, conn: sqlite3.Connection) -> dict[str, Any]:
    tables = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            df.columns = [
                re.sub(r"[^a-z0-9_]", "_", c.lower().strip()) for c in df.columns
            ]
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