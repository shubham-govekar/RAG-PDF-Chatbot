from typing import Optional

from src.state import GraphState
from src.llm_clients import get_fast_llm, get_big_llm
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

## Global Retriever Setup

# --- Embeddings & Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="rag_data",
    embedding_function=embeddings,
    persist_directory="./chroma_db_data"
)

# --- Parent/Child Docstore Setup ---
# Chunk sizes must align with the ingestion configuration.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

fs = LocalFileStore("./parent_docs")
docstore = create_kv_docstore(fs)

# Resolves granular child chunk matches back to their broader parent documents
# to provide expanded context during generation.
parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10}
)

# --- BM25 (Keyword) Retriever ---
_bm25_retriever: Optional[BM25Retriever] = None
_bm25_doc_count: int = -1


def _get_bm25_retriever() -> Optional[BM25Retriever]:
    """
    Lazily initializes the BM25 retriever and rebuilds it if the 
    Chroma collection document count changes.
    """
    global _bm25_retriever, _bm25_doc_count

    collection = vector_store.get()
    current_count = len(collection.get("documents", []))

    if current_count == 0:
        return None

    if _bm25_retriever is None or current_count != _bm25_doc_count:
        _bm25_retriever = BM25Retriever.from_texts(
            collection["documents"],
            metadatas=collection.get("metadatas"),
        )
        _bm25_retriever.k = 10
        _bm25_doc_count = current_count

    return _bm25_retriever


def _get_ensemble_retriever():
    """
    Constructs a hybrid retriever combining vector search (ParentDocumentRetriever) 
    and keyword search (BM25). Falls back to vector search if BM25 is uninitialized.
    """
    bm25 = _get_bm25_retriever()
    if bm25 is None:
        return parent_retriever

    return EnsembleRetriever(
        retrievers=[parent_retriever, bm25], weights=[0.5, 0.5]
    )


def _list_available_sources() -> list[str]:
    """
    Enumerates distinct source metadata values across all parent documents.
    """
    sources = set()
    for key in fs.yield_keys():
        doc = docstore.mget([key])[0]
        if doc is not None:
            source = doc.metadata.get("source")
            if source:
                sources.add(source)
    return sorted(sources)


def list_available_sources() -> list[str]:
    """Public wrapper to fetch available document sources for the UI."""
    return _list_available_sources()


def _fetch_parent_docs_by_source(source: str) -> list[Document]:
    """
    Retrieves all parent documents from the docstore matching the specified source.
    """
    matches = []
    for key in fs.yield_keys():
        doc = docstore.mget([key])[0]
        if doc is not None and doc.metadata.get("source") == source:
            matches.append(doc)
    return matches


# --- Rerank Compressor & LLMs ---
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
fast_llm = get_fast_llm(temperature=0)      
big_llm = get_big_llm(temperature=0.1)      

# --- Prompt for query reformulation ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriting tool, not a question-answering assistant. "
        "Your ONLY job is to rephrase the follow-up question into a standalone "
        "search query for a vector database, using context from the chat history.\n\n"
        "STRICT RULES:\n"
        "- Do NOT answer the question.\n"
        "- Do NOT explain, define, or expand any acronym or term in the question.\n"
        "- Do NOT add any information that is not already present in the chat "
        "history or the question itself.\n"
        "- Output ONLY the rewritten search query text. No quotes, no preamble, "
        "no conversational filler, no more than one sentence.\n\n"
        "Example of what NOT to do:\n"
        "Follow-up Question: What is MBO?\n"
        "WRONG output: 'MBO stands for Multi-Branch Optimization, a technique...' "
        "(this answers the question instead of rewriting it)\n"
        "CORRECT output: 'MBO algorithm'"
    ),
    ("human", "Chat History:\n{chat_history}\n\nFollow-up Question: {current_question}")
])
rewriter_chain = rewrite_prompt | fast_llm | StrOutputParser()

# --- Prompt for generation ---
generate_prompt = ChatPromptTemplate.from_template(
    """
    You are a strict document-only assistant. You must answer using ONLY the
    information inside the Context section below.

    You are NOT permitted to use any outside knowledge, even if you recognize
    the topic or believe you already know the answer. Do not guess. Do not
    fill in gaps from memory. Ignore anything you know about this subject
    that is not explicitly stated in the Context.

    If the Context does not contain enough information to answer the
    Question, you MUST respond with exactly this sentence and nothing else:
    "I cannot find the answer in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer (using ONLY the Context above):
    """
)
generation_chain = generate_prompt | big_llm | StrOutputParser()

# --- Prompt for intent classification ---
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Classify the user's query into exactly one label: qa, summary, or "
        "chitchat.\n"
        "- qa: asks a specific factual question answerable from a passage\n"
        "- summary: asks to summarize, give an overview of, or explain what "
        "a document is about, as a whole\n"
        "- chitchat: greetings, thanks, or anything unrelated to the documents\n"
        "Reply with ONLY the single label — no punctuation, no explanation."
    ),
    ("human", "{query}"),
])
intent_chain = intent_prompt | fast_llm | StrOutputParser()
_VALID_INTENTS = {"qa", "summary", "chitchat"}

# --- Prompts for summary path ---
SINGLE_SHOT_CHAR_BUDGET = 24000  # Threshold for single-call summarization
BATCH_SIZE = 5                   # Parent chunks per map call
MAX_PARENT_DOCS = 40             # Hard limit for batch processing

single_shot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the following document in 5-8 sentences. Preserve key "
        "facts, numbers, and names exactly. Do not add information that "
        "isn't in the text."
    ),
    ("human", "{document}"),
])
single_shot_chain = single_shot_prompt | big_llm | StrOutputParser()

# Map-reduce fallback for documents exceeding the single-shot budget.
map_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the following document section in 3-5 bullet points. "
        "Preserve key facts, numbers, and names exactly. Do not add "
        "information that isn't in the text."
    ),
    ("human", "{chunk}"),
])
map_chain = map_prompt | big_llm | StrOutputParser()

reduce_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Combine these partial summaries into one coherent summary of the "
        "full document, 5-8 sentences, no repetition, no meta-commentary."
    ),
    ("human", "{partial_summaries}"),
])
reduce_chain = reduce_prompt | big_llm | StrOutputParser()

# --- Prompt for source resolution ---
source_match_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "The user is asking for a summary of one of the following ingested "
        "documents. Available filenames:\n{sources}\n\n"
        "Reply with ONLY the exact filename from the list that best matches "
        "the user's request. If nothing clearly matches, reply with the "
        "first filename in the list."
    ),
    ("human", "{query}"),
])
source_match_chain = source_match_prompt | fast_llm | StrOutputParser()

# --- Prompt for chitchat ---
chitchat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly assistant for a document Q&A chatbot. Respond "
        "briefly and naturally to greetings, thanks, or small talk. If the "
        "user seems to want document help, gently prompt them to ask a "
        "question about their documents."
    ),
    ("human", "{query}"),
])
chitchat_chain = chitchat_prompt | fast_llm | StrOutputParser()


def detect_intent_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Classifies the raw query to route the graph execution path.
    """
    raw_query = state.get("raw_query", "")
    raw_intent = intent_chain.invoke({"query": raw_query}).strip().lower()

    intent = raw_intent if raw_intent in _VALID_INTENTS else "qa"

    return {"intent": intent}


def reformulate_query_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Analyzes chat history and reformulates the raw query into an optimized
    search query for the vector database.
    """
    raw_query = state.get("raw_query")
    messages = state.get("messages", [])

    if len(messages) <= 1:
        return {"search_query": raw_query}

    chat_history_str = get_buffer_string(messages[:-1])
    current_question = messages[-1].content if hasattr(messages[-1], 'content') else raw_query

    search_query = rewriter_chain.invoke({
        "chat_history": chat_history_str,
        "current_question": current_question
    })

    # Fallback to the original question if the rewrite output is excessively long
    MAX_REWRITE_WORDS = 25
    if len(search_query.split()) > MAX_REWRITE_WORDS:
        search_query = current_question

    return {"search_query": search_query}


def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Retrieves relevant document chunks using the ensemble retriever.
    """
    search_query = state.get("search_query")

    if not search_query:
        search_query = state.get("raw_query")

    ensemble_retriever = _get_ensemble_retriever()
    results = ensemble_retriever.invoke(search_query)

    target_source = state.get("target_source")
    if target_source:
        results = [d for d in results if d.metadata.get("source") == target_source]

    return {"documents": results}


def grade_documents_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Filters retrieved documents using a cross-encoder based on a relevance threshold.
    """
    documents = state.get("documents", [])
    search_query = state.get("search_query", "")

    configurable = (config or {}).get("configurable", {})
    threshold = configurable.get("relevance_threshold", 0.2)

    if not documents:
        return {"documents": []}

    compressed_docs = compressor.compress_documents(documents, search_query)

    filtered_docs = [
        doc for doc in compressed_docs
        if doc.metadata.get("relevance_score", 0.0) >= threshold
    ]

    return {"documents": filtered_docs}


def generate_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Generates an answer based on the retrieved and filtered documents.
    """
    documents = state.get("documents", [])
    raw_query = state.get("raw_query")

    context = "\n\n".join(doc.page_content for doc in documents)

    response_string = generation_chain.invoke(
        {
            "context": context,
            "question": raw_query
        }
    )

    return {"generation": response_string}


def summarize_document_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Bypasses similarity search to summarize a target document directly from the docstore.
    Uses map-reduce chunking for documents exceeding the single-shot length budget.
    """
    raw_query = state.get("raw_query", "")
    sources = _list_available_sources()

    if not sources:
        return {"generation": "No documents have been ingested yet."}

    target_source = state.get("target_source")
    if not target_source:
        if len(sources) == 1:
            target_source = sources[0]
        else:
            target_source = source_match_chain.invoke({
                "sources": "\n".join(sources),
                "query": raw_query,
            }).strip()
            
            if target_source not in sources:
                target_source = sources[0]
    elif target_source not in sources:
        return {"generation": f"'{target_source}' isn't an ingested document."}

    parent_docs = _fetch_parent_docs_by_source(target_source)[:MAX_PARENT_DOCS]
    if not parent_docs:
        return {"generation": f"I couldn't find content for '{target_source}'."}

    full_text = "\n\n".join(doc.page_content for doc in parent_docs)

    if len(full_text) <= SINGLE_SHOT_CHAR_BUDGET:
        summary = single_shot_chain.invoke({"document": full_text})
    else:
        batches = [
            parent_docs[i:i + BATCH_SIZE]
            for i in range(0, len(parent_docs), BATCH_SIZE)
        ]
        partial_summaries = [
            map_chain.invoke({"chunk": "\n\n".join(d.page_content for d in batch)})
            for batch in batches
        ]
        summary = reduce_chain.invoke({
            "partial_summaries": "\n\n".join(partial_summaries)
        })

    return {"generation": summary, "target_source": target_source}


def chitchat_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Handles standard conversational queries to bypass vector retrieval.
    """
    response = chitchat_chain.invoke({"query": state.get("raw_query", "")})
    return {"generation": response}