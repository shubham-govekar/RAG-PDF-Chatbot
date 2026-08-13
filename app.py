from dotenv import load_dotenv
load_dotenv()

import os

if not os.path.exists("chroma_db_data") or not os.listdir("chroma_db_data"):
    from offline_ingestion import run_ingestion
    run_ingestion()

import logging

import streamlit as st
from src.graph import app  # Your compiled graph
from src.nodes import list_available_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Local RAG Assistant", layout="wide")


def _render_sources(documents, search_query=None, intent=None, target_source=None):
    """Displays retrieval details for the QA path, or a short note for the
    summary/chitchat paths, which don't run similarity search at all."""
    if intent == "summary":
        st.caption(f"Summarized directly from `{target_source or 'unknown'}` (no similarity search).")
        return
    if intent == "chitchat":
        st.caption("No retrieval performed — chitchat response.")
        return

    if search_query:
        st.markdown(f"**Search query used:** `{search_query}`")

    if not documents:
        if target_source:
            st.warning(f"No relevant content found in the scoped document ('{target_source}').")
        else:
            st.warning("No documents passed the relevance threshold.")
        return

    for i, doc in enumerate(documents, start=1):
        score = doc.metadata.get("relevance_score")
        score_str = f"{score:.3f}" if score is not None else "n/a"
        source = doc.metadata.get("source", "unknown source")
        st.markdown(f"**[{i}] score: {score_str} — {source}**")
        preview = doc.page_content[:500]
        suffix = "..." if len(doc.page_content) > 500 else ""
        st.text(preview + suffix)
        st.divider()


# --- Sidebar: adjustable retrieval settings ---
with st.sidebar:
    st.header("Retrieval Settings")
    relevance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Documents scoring below this after reranking are dropped. "
             "Raise this if the model seems to be answering from weak/irrelevant context."
    )
    st.caption(
        "Lower = more documents pass through (higher recall, more noise). "
        "Higher = stricter (higher precision, more 'I cannot find the answer' results)."
    )
    st.caption("Only applies to the QA path — summary and chitchat queries skip retrieval.")

    st.header("Document Scope")
    scope_options = ["All documents"] + list_available_sources()
    selected_source = st.selectbox(
        "Limit to a specific document (optional)",
        scope_options,
        help="Applies to both QA and summary. Leave on 'All documents' to "
             "let retrieval (QA) or auto-detection (summary) decide.",
    )
    target_source = None if selected_source == "All documents" else selected_source

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Details"):
                _render_sources(
                    message.get("sources") or [],
                    message.get("search_query"),
                    message.get("intent"),
                    message.get("target_source"),
                )

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare inputs for the graph
    inputs = {
        "messages": st.session_state.messages,
        "raw_query": prompt,
        "target_source": target_source,
    }
    config = {"configurable": {"relevance_threshold": relevance_threshold}}

    result = {}
    generation = "I'm sorry, I couldn't find an answer."

    with st.chat_message("assistant"):
        # NOTE: label kept generic since the graph now branches into three
        # paths (qa / summary / chitchat) before we know which one ran.
        with st.status("Thinking...", expanded=False) as status:
            try:
                result = app.invoke(inputs, config=config)
                status.update(label="Done", state="complete")
            except Exception:
                logger.exception("Error while generating a response")
                status.update(label="Failed", state="error")
                st.error(
                    "An internal error occurred. Please try again or check "
                    "server logs."
                )

        generation = result.get("generation", generation)
        st.markdown(generation)

        # Surface exactly what was retrieved and used, so hallucinations can
        # be diagnosed: if the right chunks are here but the answer is still
        # wrong, it's a prompt/model-adherence issue. If the chunks are
        # missing or irrelevant, it's a retrieval/threshold issue.
        documents = result.get("documents", [])
        search_query = result.get("search_query")
        intent = result.get("intent")
        target_source = result.get("target_source")

        with st.expander("Details"):
            _render_sources(documents, search_query, intent, target_source)

    # Persist to history, including sources so they can be re-shown on rerun
    st.session_state.messages.append({
        "role": "assistant",
        "content": generation,
        "sources": documents,
        "search_query": search_query,
        "intent": intent,
        "target_source": target_source,
    })