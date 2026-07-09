from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of RAG pipeline
    """

    # Chat history (automatically appended to, not overwritten)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The original question asked by the user
    raw_query: str
    # The LLM-reformulated query used for the actual vector search
    search_query: str
    # The filtered, high-relevance chunks retrieved from Chroma/BM25
    documents: Sequence[Document]
    # The final streamed response from the cloud LLM
    generation: str

    # NEW: set by detect_intent_node — "qa", "summary", or "chitchat".
    # Drives the conditional edge in graph.py that routes each query to the
    # QA path, the summary path, or the chitchat path.
    intent: str
    # NEW: set by summarize_document_node — the resolved source filename
    # the summary was generated from. Not read by any other node; carried
    # in state mainly for debugging/eval logging (e.g. confirming the
    # right document was picked for a given summary request).
    target_source: str

    # NOTE: Dynamic UI settings (e.g. relevance_threshold) are no longer
    # carried as a field on this state. They're passed via LangGraph's native
    # `config={"configurable": {...}}` argument to app.invoke(), which nodes
    # receive automatically as their second argument. This keeps the
    # "configurable" concept consistent with LangGraph's own terminology
    # instead of overloading a custom state key with the same name.