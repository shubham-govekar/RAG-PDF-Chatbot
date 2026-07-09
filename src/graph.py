from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes import (
    detect_intent_node,
    reformulate_query_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    summarize_document_node,
    chitchat_node,
)

# Initialize the workflow with the state schema
workflow = StateGraph(GraphState)

# Add execution nodes
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("reformulate", reformulate_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("summarize_document", summarize_document_node)
workflow.add_node("chitchat", chitchat_node)

# Set entry point
workflow.set_entry_point("detect_intent")


def route_by_intent(state):
    """
    Routes the query to the appropriate execution path based on the detected intent:
      - "qa": Standard retrieval-augmented generation.
      - "summary": Direct document summarization (bypasses retrieval).
      - "chitchat": Conversational response (bypasses retrieval).
    """
    return state.get("intent", "qa")


workflow.add_conditional_edges(
    "detect_intent",
    route_by_intent,
    {
        "qa": "reformulate",
        "summary": "summarize_document",
        "chitchat": "chitchat",
    },
)

# QA execution path
workflow.add_edge("reformulate", "retrieve")
workflow.add_edge("retrieve", "grade")


def decide_to_generate(state):
    """
    Determines whether to proceed to generation based on retrieved documents.
    Ends the workflow if no relevant documents passed the grading stage.
    """
    if not state.get("documents"):
        return "end"
    return "generate"


workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "end": END
    }
)

# Finalize all execution paths
workflow.add_edge("generate", END)
workflow.add_edge("summarize_document", END)
workflow.add_edge("chitchat", END)

# Compile the graph
app = workflow.compile()