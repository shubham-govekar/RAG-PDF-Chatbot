from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.graph import app  
from src.nodes import list_available_sources

st.set_page_config(page_title="Local RAG Assistant", layout="wide")


def _render_sources(documents, search_query=None, intent=None, target_source=None):
    """
    Renders retrieval metadata or execution path details in the UI expander.
    Handles conditionally bypassed retrieval for summary and chitchat intents.
    """
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


# --- Sidebar: Retrieval Configuration ---
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
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

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Construct state payload and configuration for graph execution
    inputs = {
        "messages": st.session_state.messages,
        "raw_query": prompt,
        "target_source": target_source,
    }
    config = {"configurable": {"relevance_threshold": relevance_threshold}}

    result = {}
    generation = "I'm sorry, I couldn't find an answer."

    with st.chat_message("assistant"):
        
        # Execute graph with generic status indicator (intent routing occurs dynamically)
        with st.status("Thinking...", expanded=False) as status:
            try:
                result = app.invoke(inputs, config=config)
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Something went wrong while generating a response: {e}")

        generation = result.get("generation", generation)
        st.markdown(generation)

        # Extract execution metadata to populate UI details for diagnostic evaluation
        documents = result.get("documents", [])
        search_query = result.get("search_query")
        intent = result.get("intent")
        target_source = result.get("target_source")

        with st.expander("Details"):
            _render_sources(documents, search_query, intent, target_source)

    # Persist response and metadata to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": generation,
        "sources": documents,
        "search_query": search_query,
        "intent": intent,
        "target_source": target_source,
    })