import json
import re
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def generate_sql(
    client: Groq,
    messages: list[dict],
    model: str = GROQ_MODEL,
) -> tuple[str, str]:
    """
    Returns (sql, explanation).
    Raises ValueError if response cannot be parsed.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_response(raw)


def generate_sql_retry(
    client: Groq,
    messages: list[dict],
    bad_sql: str,
    error: str,
    model: str = GROQ_MODEL,
) -> tuple[str, str]:
    """
    Called after a failed SQL execution. Sends the error back to the LLM.
    Returns (sql, explanation).
    """
    retry_messages = messages + [
        {
            "role": "assistant",
            "content": json.dumps({"sql": bad_sql, "explanation": ""}),
        },
        {
            "role": "user",
            "content": (
                f"That SQL failed with this SQLite error:\n{error}\n\n"
                "Please fix the SQL and return the corrected JSON response only."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=retry_messages,
        temperature=0,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_response(raw)


def _parse_response(raw: str) -> tuple[str, str]:
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        sql = parsed.get("sql", "").strip()
        explanation = parsed.get("explanation", "").strip()
        if sql:
            return sql, explanation
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON object via regex
    match = re.search(r'\{.*"sql"\s*:.*\}', cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            sql = parsed.get("sql", "").strip()
            explanation = parsed.get("explanation", "").strip()
            if sql:
                return sql, explanation
        except json.JSONDecodeError:
            pass

    # Last resort: extract anything that looks like a SELECT statement
    sql_match = re.search(r"(SELECT\s.+?)(?:```|$)", cleaned, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip(), ""

    raise ValueError(f"Could not parse LLM response as SQL JSON.\nRaw response:\n{raw}")