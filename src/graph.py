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

# 1. Initialize the Graph with our State schema
workflow = StateGraph(GraphState)

# 2. Add the Nodes
# NOTE: each node function now has the signature (state, config). LangGraph
# detects this automatically and passes the RunnableConfig (which carries
# config["configurable"]["relevance_threshold"] from app.py) as the second
# argument — no changes needed here beyond the node functions themselves.
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("reformulate", reformulate_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("summarize_document", summarize_document_node)
workflow.add_node("chitchat", chitchat_node)

# 3. Define the Flow (The Edges)
# NEW: detect_intent now runs first, before reformulation — reformulation
# only makes sense for the QA path, so it shouldn't run unconditionally.
workflow.set_entry_point("detect_intent")


def route_by_intent(state):
    """
    NEW: routes each query to one of three paths based on detect_intent_node's
    classification:
      - "qa"       -> reformulate -> retrieve -> grade -> generate (unchanged)
      - "summary"  -> summarize_document (bypasses retrieval entirely — see
                       summarize_document_node's docstring for why)
      - "chitchat" -> chitchat (skips retrieval; avoids the QA path's
                       grounding fallback misfiring on small talk)
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

# Existing QA path, unchanged:
workflow.add_edge("reformulate", "retrieve")
workflow.add_edge("retrieve", "grade")


# 4. Conditional Edge: The Brain
def decide_to_generate(state):
    """Determines whether to generate an answer or end the conversation."""
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

# 5. Final Edges — all three paths converge and end here.
workflow.add_edge("generate", END)
workflow.add_edge("summarize_document", END)
workflow.add_edge("chitchat", END)

# 6. Compile
app = workflow.compile()