import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("prompt_logs.jsonl")

def log_prompt(question: str, messages: list, sql: str, explanation: str, error: str = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "messages_sent_to_llm": messages,
        "sql_generated": sql,
        "explanation": explanation,
        "error": error,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")