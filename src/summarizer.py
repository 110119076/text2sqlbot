from groq import Groq


SUMMARIZE_PROMPT = """You are summarizing a data analysis conversation for memory compression.
Given the conversation turns below, produce a compact summary that preserves:
- Which tables and columns were referenced
- Filters, conditions, or time ranges the user mentioned
- Any clarifications or corrections the user made
- The general analytical direction of the conversation

Be concise. Use bullet points. Max 150 words."""


def maybe_summarize(
    history: list[dict],
    client: Groq,
    model: str,
    max_full_turns: int = 6,
    keep_recent: int = 3,
) -> list[dict]:
    """
    history is a flat list of {"role": "user"|"assistant", "content": str}
    Returns a (possibly compressed) history list.
    """
    # Count actual turns (user+assistant pairs)
    turn_count = sum(1 for m in history if m["role"] == "user")

    if turn_count <= max_full_turns:
        return history

    # Split: old part to summarize, recent part to keep in full
    # Each turn = 1 user + 1 assistant message = 2 items
    recent_items = keep_recent * 2
    old_history = history[:-recent_items]
    recent_history = history[-recent_items:]

    if not old_history:
        return history

    # Build conversation text for summarization
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_history
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": convo_text},
            ],
            temperature=0,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        summary_message = {
            "role": "assistant",
            "content": f"[Conversation summary so far]\n{summary}",
        }
        return [summary_message] + recent_history
    except Exception:
        # On failure, just trim old history and keep recent
        return recent_history