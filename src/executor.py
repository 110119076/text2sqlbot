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
    Full pipeline: generate SQL → execute → retry on error OR empty result.

    Retry triggers:
      - SQLite execution error → send error back to LLM
      - Empty result (0 rows) → send sample rows back to LLM so it can fix
        column value casing / filters

    Returns a dict:
    {
        "sql": str,
        "explanation": str,
        "df": pd.DataFrame | None,
        "total_rows": int,
        "display_rows": int,
        "error": str | None,
        "retried": bool,
        "empty_retried": bool,
    }
    """
    # Attempt 1: generate + execute
    try:
        sql, explanation = generate_sql(client, messages, model)
    except ValueError as e:
        return _error_result("", str(e))

    exec_result = _try_execute(conn, sql)

    if not exec_result["success"]:
        # Attempt 2a: SQL error retry
        try:
            fixed_sql, fixed_explanation = generate_sql_retry(
                client, messages, sql, exec_result["error"], model
            )
        except ValueError as e:
            return _error_result(sql, str(e), retried=True)

        exec_result2 = _try_execute(conn, fixed_sql)
        if exec_result2["success"]:
            return _build_result(
                fixed_sql, fixed_explanation or explanation,
                exec_result2["df"], retried=True
            )
        return _error_result(
            fixed_sql,
            f"Attempt 1 error: {exec_result['error']}\n"
            f"Attempt 2 error: {exec_result2['error']}",
            retried=True,
        )

    df = exec_result["df"]

    # Attempt 2b: Empty result retry
    # A valid but empty result often means wrong column value casing or a bad
    # WHERE filter. Send sample data back so the LLM can self-correct.
    if len(df) == 0:
        sample_hint = _get_sample_hint(conn, sql)
        empty_feedback = (
            f"The SQL executed successfully but returned 0 rows:\n{sql}\n\n"
            f"{sample_hint}\n\n"
            "The zero-row result is likely due to a wrong column value, incorrect "
            "casing in a WHERE clause, or a filter that doesn't match the actual data. "
            "Rewrite the SQL to return meaningful results. "
            "Return the corrected JSON response only."
        )
        try:
            fixed_sql, fixed_explanation = generate_sql_retry(
                client, messages, sql, empty_feedback, model
            )
        except ValueError:
            # If retry also fails to parse, return the 0-row result as-is
            return _build_result(sql, explanation, df, empty_retried=True)

        exec_result3 = _try_execute(conn, fixed_sql)
        if exec_result3["success"] and len(exec_result3["df"]) > 0:
            return _build_result(
                fixed_sql, fixed_explanation or explanation,
                exec_result3["df"], empty_retried=True
            )
        # If still empty or failed, return original 0-row result — don't hide it
        return _build_result(sql, explanation, df, empty_retried=True)

    return _build_result(sql, explanation, df)


def _get_sample_hint(conn: sqlite3.Connection, sql: str) -> str:
    """
    Extract table names from the SQL and return a few sample rows from each
    to help the LLM understand actual data values and casing.
    """
    import re
    tables_mentioned = re.findall(
        r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.IGNORECASE
    )
    flat = [t for pair in tables_mentioned for t in pair if t]
    hints = []
    for table in flat[:3]:
        try:
            sample_df = pd.read_sql_query(
                f"SELECT * FROM {table} LIMIT 3", conn
            )
            hints.append(f"Sample rows from '{table}':\n{sample_df.to_string(index=False)}")
        except Exception:
            pass
    return "\n\n".join(hints) if hints else "Could not retrieve sample rows."


def _build_result(
    sql: str,
    explanation: str,
    df: pd.DataFrame,
    retried: bool = False,
    empty_retried: bool = False,
) -> dict:
    total = len(df)
    return {
        "sql": sql,
        "explanation": explanation,
        "df": df.head(MAX_DISPLAY_ROWS),
        "total_rows": total,
        "display_rows": min(total, MAX_DISPLAY_ROWS),
        "error": None,
        "retried": retried,
        "empty_retried": empty_retried,
    }


def _try_execute(conn: sqlite3.Connection, sql: str) -> dict:
    try:
        df = pd.read_sql_query(sql, conn)
        return {"success": True, "df": df, "error": None}
    except Exception as e:
        return {"success": False, "df": None, "error": str(e)}


def _error_result(sql: str, error: str, retried: bool = False) -> dict:
    return {
        "sql": sql,
        "explanation": "",
        "df": None,
        "total_rows": 0,
        "display_rows": 0,
        "error": error,
        "retried": retried,
        "empty_retried": False,
    }