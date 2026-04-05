SYSTEM_PROMPT_TEMPLATE = """You are an expert SQLite SQL assistant. Users upload CSV files that become SQLite tables, and you help them query their data.

AVAILABLE TABLES (relevant to this question):
{schema_section}

{relationships_section}

STRICT RULES:
- Write only SELECT statements — never INSERT, UPDATE, DELETE, DROP, or ALTER
- Use ONLY the table names and column names listed above — do not invent columns
- Use standard SQLite syntax (e.g., strftime for dates, not DATE_FORMAT)
- Always alias aggregations for clarity (e.g., SUM(amount) AS total_amount)
- For ambiguous column names in JOINs, always qualify with table name

OUTPUT FORMAT:
Respond with a single valid JSON object and nothing else — no markdown, no explanation outside JSON:
{{
  "sql": "<your SQL query here>",
  "explanation": "<1 to 3 sentences explaining what the query does in plain English>"
}}"""


def build_schema_section(relevant_tables: dict) -> str:
    lines = []
    for table_name, meta in relevant_tables.items():
        cols = ", ".join(
            f"{c['name']} ({c['type']})" for c in meta["columns"]
        )
        lines.append(f"TABLE: {table_name}")
        lines.append(f"  Columns: {cols}")
        lines.append(f"  Row count: {meta['row_count']:,}")
        lines.append(f"  Sample rows:\n{_indent(meta['sample_rows'], 4)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_relationships_section(relationships: list[dict], relevant_tables: dict) -> str:
    if not relationships:
        return ""
    relevant_names = set(relevant_tables.keys())
    relevant_rels = [
        r for r in relationships
        if r["table_a"] in relevant_names and r["table_b"] in relevant_names
    ]
    if not relevant_rels:
        return ""
    lines = ["DETECTED RELATIONSHIPS (suggested JOIN keys):"]
    for r in relevant_rels:
        lines.append(
            f"  {r['table_a']}.{r['col_a']} → {r['table_b']}.{r['col_b']}"
        )
    return "\n".join(lines)


def build_messages(
    question: str,
    relevant_tables: dict,
    relationships: list[dict],
    history: list[dict],
) -> list[dict]:
    schema_section = build_schema_section(relevant_tables)
    relationships_section = build_relationships_section(relationships, relevant_tables)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        schema_section=schema_section,
        relationships_section=relationships_section,
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())