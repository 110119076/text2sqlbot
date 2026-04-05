import sqlite3
import pandas as pd
from groq import Groq

from src.llm import generate_sql, generate_sql_retry, GROQ_MODEL

MAX_DISPLAY_ROWS = 100


def execute_with_retry(
    client: Groq,
    messages: list[dict],
    conn: sqlite3.Connection,
    model: str = GROQ_MODEL,
) -> dict:
    """
    Full pipeline: generate SQL → execute → retry once on failure.

    Returns a dict:
    {
        "sql": str,
        "explanation": str,
        "df": pd.DataFrame | None,
        "total_rows": int,
        "display_rows": int,
        "error": str | None,
        "retried": bool,
    }
    """
    # Attempt 1
    try:
        sql, explanation = generate_sql(client, messages, model)
    except ValueError as e:
        return _error_result("", str(e), retried=False)

    result = _try_execute(conn, sql)

    if result["success"]:
        df = result["df"]
        total = len(df)
        return {
            "sql": sql,
            "explanation": explanation,
            "df": df.head(MAX_DISPLAY_ROWS),
            "total_rows": total,
            "display_rows": min(total, MAX_DISPLAY_ROWS),
            "error": None,
            "retried": False,
        }

    # Attempt 2 — send error back to LLM
    try:
        fixed_sql, fixed_explanation = generate_sql_retry(
            client, messages, sql, result["error"], model
        )
    except ValueError as e:
        return _error_result(sql, str(e), retried=True)

    result2 = _try_execute(conn, fixed_sql)

    if result2["success"]:
        df = result2["df"]
        total = len(df)
        return {
            "sql": fixed_sql,
            "explanation": fixed_explanation or explanation,
            "df": df.head(MAX_DISPLAY_ROWS),
            "total_rows": total,
            "display_rows": min(total, MAX_DISPLAY_ROWS),
            "error": None,
            "retried": True,
        }

    # Both attempts failed
    return _error_result(
        fixed_sql,
        f"Attempt 1: {result['error']}\nAttempt 2: {result2['error']}",
        retried=True,
    )


def _try_execute(conn: sqlite3.Connection, sql: str) -> dict:
    try:
        df = pd.read_sql_query(sql, conn)
        return {"success": True, "df": df, "error": None}
    except Exception as e:
        return {"success": False, "df": None, "error": str(e)}


def _error_result(sql: str, error: str, retried: bool) -> dict:
    return {
        "sql": sql,
        "explanation": "",
        "df": None,
        "total_rows": 0,
        "display_rows": 0,
        "error": error,
        "retried": retried,
    }