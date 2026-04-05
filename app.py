import os
import sqlite3

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import load_csvs, infer_relationships
from src.retriever import build_schema_embeddings, retrieve_relevant_tables
from src.prompt_builder import build_messages
from src.summarizer import maybe_summarize
from src.executor import execute_with_retry
from src.llm import get_groq_client, GROQ_MODEL

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV - SQL Assistant",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a0d13;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none; }

/* Custom top bar */
.top-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 0 24px 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 24px;
}
.top-bar h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: #7dd3fc;
    margin: 0;
    letter-spacing: -0.02em;
}
.top-bar .badge {
    background: #0ea5e920;
    border: 1px solid #0ea5e940;
    color: #7dd3fc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Chat messages */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 8px 0;
}
.chat-user-bubble {
    background: #1e3a5f;
    border: 1px solid #2563eb40;
    color: #e2e8f0;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
}
.chat-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0 16px 0;
}
.chat-assistant-content {
    max-width: 95%;
}

/* SQL block header */
.sql-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    margin-top: 12px;
}
.sql-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.retry-badge {
    background: #451a03;
    border: 1px solid #92400e;
    color: #fbbf24;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 1px 6px;
    border-radius: 3px;
    letter-spacing: 0.06em;
}

/* Explanation block */
.explanation-block {
    background: #0f1f35;
    border-left: 3px solid #0ea5e9;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin-top: 10px;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* Row count pill */
.row-count {
    display: inline-block;
    background: #0f2027;
    border: 1px solid #1e3a5f;
    color: #7dd3fc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}

/* Error block */
.error-block {
    background: #1c0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 6px;
    padding: 12px 16px;
    color: #fca5a5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* Table chip in sidebar */
.table-chip {
    background: #0f1f35;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.table-chip .tname {
    font-family: 'IBM Plex Mono', monospace;
    color: #7dd3fc;
    font-size: 0.83rem;
}
.table-chip .tmeta {
    color: #475569;
    font-size: 0.75rem;
    margin-top: 2px;
}

/* Relationship chip */
.rel-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #a78bfa;
    background: #1a1040;
    border: 1px solid #3b1f7a40;
    padding: 4px 10px;
    border-radius: 4px;
    margin-bottom: 4px;
    display: block;
}

/* Divider */
.chat-divider {
    border: none;
    border-top: 1px solid #1e2535;
    margin: 8px 0;
}

/* Input area */
.stChatInput > div {
    background: #0a0d13 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 12px !important;
}

/* Scrollable chat area */
.chat-scroll {
    max-height: 68vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* Welcome state */
.welcome-box {
    text-align: center;
    padding: 60px 20px;
    color: #334155;
}
.welcome-box h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: #475569;
    margin-bottom: 12px;
}
.welcome-box p {
    font-size: 0.9rem;
    color: #334155;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.7;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
    margin-top: 16px;
}

/* Dataframe override */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2535 !important;
    border-radius: 6px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# Session state init
def init_state():
    defaults = {
        "conn": None,
        "tables": {},
        "relationships": [],
        "schema_embeddings": {},
        "chat_history": [],      # full message history for LLM
        "display_messages": [],  # what we render in UI
        "loaded_files": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# Helpers
def get_client():
    api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    if not api_key:
        return None
    return get_groq_client(api_key)


def reset_session():
    keys = ["conn", "tables", "relationships", "schema_embeddings",
            "chat_history", "display_messages", "loaded_files"]
    for k in keys:
        del st.session_state[k]
    init_state()


# Sidebar
with st.sidebar:
    st.markdown('<div class="section-label">API Key</div>', unsafe_allow_html=True)
    if not os.getenv("GROQ_API_KEY"):
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if api_key_input:
            st.session_state["groq_api_key"] = api_key_input
    else:
        st.markdown(
            '<span style="color:#4ade80;font-size:0.8rem;">✓ API key loaded from .env</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top:20px;">Upload CSVs</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_names = sorted([f.name for f in uploaded_files])
        if new_names != sorted(st.session_state.get("loaded_files", [])):
            with st.spinner("Loading tables..."):
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                tables = load_csvs(uploaded_files, conn)
                relationships = infer_relationships(tables)
                schema_embeddings = build_schema_embeddings(tables)

                st.session_state.conn = conn
                st.session_state.tables = tables
                st.session_state.relationships = relationships
                st.session_state.schema_embeddings = schema_embeddings
                st.session_state.loaded_files = new_names
                st.session_state.chat_history = []
                st.session_state.display_messages = []

    # Tables list
    if st.session_state.tables:
        st.markdown('<div class="section-label" style="margin-top:20px;">Loaded Tables</div>', unsafe_allow_html=True)
        for tname, meta in st.session_state.tables.items():
            cols_preview = ", ".join(c["name"] for c in meta["columns"][:4])
            if len(meta["columns"]) > 4:
                cols_preview += f" +{len(meta['columns'])-4} more"
            st.markdown(
                f"""<div class="table-chip">
                    <div class="tname">⬡ {tname}</div>
                    <div class="tmeta">{meta['row_count']:,} rows · {cols_preview}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Relationships
        if st.session_state.relationships:
            st.markdown('<div class="section-label" style="margin-top:16px;">Detected Joins</div>', unsafe_allow_html=True)
            for rel in st.session_state.relationships:
                st.markdown(
                    f'<span class="rel-chip">{rel["table_a"]}.{rel["col_a"]} → {rel["table_b"]}.{rel["col_b"]}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear & Reset", use_container_width=True):
            reset_session()
            st.rerun()


# Main area
st.markdown(
    """<div class="top-bar">
        <h1>🗃️ CSV - SQL Assistant</h1>
        <span class="badge">Groq · LLaMA 3.3</span>
    </div>""",
    unsafe_allow_html=True,
)

# Chat display
chat_container = st.container()

with chat_container:
    if not st.session_state.display_messages:
        st.markdown(
            """<div class="welcome-box">
                <h2>Upload CSVs to get started</h2>
                <p>Upload one or more CSV files using the sidebar, then ask questions in plain English. The assistant will generate SQL and return your results.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.display_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                res = msg["result"]
                st.markdown('<div class="chat-assistant"><div class="chat-assistant-content">', unsafe_allow_html=True)

                # SQL block
                retry_badge = '<span class="retry-badge">RETRIED</span>' if res.get("retried") else ""
                st.markdown(
                    f'<div class="sql-header"><span class="sql-label">Generated SQL</span>{retry_badge}</div>',
                    unsafe_allow_html=True,
                )
                if res.get("sql"):
                    st.code(res["sql"], language="sql")

                # Results or error
                if res.get("error"):
                    st.markdown(
                        f'<div class="error-block">⚠ Query failed after retry:\n{res["error"]}</div>',
                        unsafe_allow_html=True,
                    )
                elif res.get("df") is not None:
                    st.markdown(
                        f'<div class="row-count">Showing {res["display_rows"]} of {res["total_rows"]:,} rows</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(res["df"], use_container_width=True)

                # Explanation
                if res.get("explanation"):
                    st.markdown(
                        f'<div class="explanation-block">💡 {res["explanation"]}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div></div>", unsafe_allow_html=True)
                st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)


# Chat input
question = st.chat_input(
    "Ask a question about your data...",
    disabled=not st.session_state.tables,
)

if question:
    client = get_client()
    if not client:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    if not st.session_state.tables:
        st.warning("Please upload CSV files first.")
        st.stop()

    # Add user message to display
    st.session_state.display_messages.append({"role": "user", "content": question})

    with st.spinner("Generating SQL..."):
        # Retrieve relevant tables
        relevant_tables = retrieve_relevant_tables(
            question,
            st.session_state.schema_embeddings,
            st.session_state.tables,
        )

        # Maybe compress history
        compressed_history = maybe_summarize(
            st.session_state.chat_history,
            client,
            GROQ_MODEL,
        )

        # Build messages
        messages = build_messages(
            question,
            relevant_tables,
            st.session_state.relationships,
            compressed_history,
        )

        # Execute with retry
        result = execute_with_retry(
            client,
            messages,
            st.session_state.conn,
        )

    # Update LLM chat history (plain text only, no DataFrames)
    st.session_state.chat_history.append({"role": "user", "content": question})
    assistant_summary = result["sql"] if result["sql"] else result.get("error", "")
    st.session_state.chat_history.append(
        {"role": "assistant", "content": f"SQL: {assistant_summary}"}
    )

    # Add to display
    st.session_state.display_messages.append({"role": "assistant", "result": result})

    st.rerun()