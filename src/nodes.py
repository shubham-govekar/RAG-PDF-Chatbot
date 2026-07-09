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
# These splitter configs MUST match offline_ingestion.py. The child chunks
# already embedded in Chroma were created with these sizes; ParentDocumentRetriever
# uses them (plus the docstore) to map a matched child chunk back to its
# full parent document. It does not re-split anything unless add_documents()
# is called, which only happens during ingestion, not here.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

fs = LocalFileStore("./parent_docs")
docstore = create_kv_docstore(fs)

# FIX: This is the core retrieval fix. Previously the pipeline searched Chroma
# directly with `vector_store.as_retriever(...)`, which returns the raw 400-char
# child chunks used only for embedding precision. That meant the docstore built
# during ingestion (the 2000-char parent windows) was never actually read.
# ParentDocumentRetriever searches the same child embeddings in Chroma, but
# resolves each hit's doc_id back to its parent via the docstore, so
# generation gets the larger, more coherent context it was designed for.
parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10}
)

# --- BM25 (Keyword) Retriever, rebuilt when the collection changes ---
_bm25_retriever: Optional[BM25Retriever] = None
_bm25_doc_count: int = -1


def _get_bm25_retriever() -> Optional[BM25Retriever]:
    """
    Lazily builds the BM25 retriever, and rebuilds it if the number of
    documents in the Chroma collection has changed since the last build.

    FIX: Previously this was built exactly once at module import time, so if
    `offline_ingestion.py` was re-run to add new documents while the app was
    already running, BM25 results would silently go stale until a restart,
    while the vector leg of the ensemble would already reflect the new data.
    This also guards the case where the app starts before ingestion has ever
    been run (empty collection), which would otherwise throw on import.
    """
    global _bm25_retriever, _bm25_doc_count

    collection = vector_store.get()
    current_count = len(collection.get("documents", []))

    if current_count == 0:
        return None

    if _bm25_retriever is None or current_count != _bm25_doc_count:
        # FIX: previously built with texts only, which silently dropped
        # source metadata on every BM25 hit (they'd render as "unknown
        # source" in the UI, and couldn't be filtered by document scope).
        # Chroma's .get() returns "documents" and "metadatas" in matching
        # order, so this is a safe zip.
        _bm25_retriever = BM25Retriever.from_texts(
            collection["documents"],
            metadatas=collection.get("metadatas"),
        )
        _bm25_retriever.k = 10
        _bm25_doc_count = current_count

    return _bm25_retriever


def _get_ensemble_retriever():
    """
    Builds the hybrid retriever for the current request. Falls back to
    parent-doc retrieval alone if BM25 has no data yet (e.g. ingestion
    hasn't been run), instead of crashing on an empty index.
    """
    bm25 = _get_bm25_retriever()
    if bm25 is None:
        return parent_retriever

    return EnsembleRetriever(
        retrievers=[parent_retriever, bm25], weights=[0.5, 0.5]
    )


def _list_available_sources() -> list[str]:
    """
    Enumerates the distinct `source` metadata values across every parent
    document in the docstore. Used by summarize_document_node to resolve
    which ingested file a summary request refers to.

    NOTE: iterates the full docstore via fs.yield_keys(). Fine at the scale
    of a resume-project corpus (a handful of PDFs); if this ever needs to
    scale to hundreds of documents, maintain a small source->doc_ids index
    at ingestion time instead of scanning on every summary request.
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
    """Public wrapper around _list_available_sources — used by app.py to
    populate a document-scope selector in the sidebar."""
    return _list_available_sources()


def _fetch_parent_docs_by_source(source: str) -> list[Document]:
    """
    Returns every parent document whose `source` metadata exactly matches
    the given filename, straight from the docstore — bypasses similarity
    search entirely, which is the point of the summary path.
    """
    matches = []
    for key in fs.yield_keys():
        doc = docstore.mget([key])[0]
        if doc is not None and doc.metadata.get("source") == source:
            matches.append(doc)
    return matches


# --- Rerank Compressor & LLMs ---
# FIX: These were previously instantiated fresh inside their node functions
# on every single graph run. FlashrankRerank in particular reloads a
# cross-encoder model from disk each time, which is expensive. Hoisting them
# to module scope means they're loaded once per process.
#
# UPDATE: generation and reformulation moved off local Ollama (qwen2.5:1.5b)
# onto cloud-hosted open-weight models via src/llm_clients.py — see that
# module for the Groq/NVIDIA NIM setup and model choice notes. FlashRank and
# the retrievers above stay local/unchanged; they were never the source of
# the latency or hallucination problems.
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
fast_llm = get_fast_llm(temperature=0)      # reformulate_query, detect_intent, chitchat
big_llm = get_big_llm(temperature=0.1)      # generate, summarize_document

# --- Prompt for query reformulation ---
# FIX (hallucination, same root cause as generate_prompt): the model was
# ignoring "Do NOT answer the question" and answering anyway, e.g. inventing
# "MBO stands for Multi-Branch Optimization" instead of just rewriting the
# query. A single negative instruction buried in a sentence is easy for a
# 1.5B model to drop. Made blunt, repeated, and anchored with an example of
# the failure mode to avoid.
#
# NOTE: kept intentionally as-is after the model swap. The hosted models are
# far less prone to this failure mode, but the length-guard fallback in
# reformulate_query_node below costs nothing to keep as a safety net.
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
# FIX (hallucination): small local models like qwen2.5:1.5b frequently ignore
# a single polite grounding instruction and fall back on their own pretrained
# knowledge when a topic feels familiar (this is exactly what produced the
# "RAG = Rich Answer Generation" answer). Small models respond far better to
# blunt, repeated, explicit instructions than to one clause buried in a
# sentence. This prompt is intentionally redundant.
#
# NOTE: kept as-is after the model swap for the same reason as above — the
# grounding instruction costs nothing and is good practice regardless of
# model size.
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
# NEW: routes each query to one of three graph paths. Similarity search
# can't answer "summarize document X" — there's no single relevant chunk to
# retrieve — so summary requests need to bypass retrieve/grade entirely and
# go straight to the full parent document(s) instead.
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
# NEW: a document under this many characters is summarized in a single
# LLM call instead of map-reduce. A typical resume-project PDF (a handful
# of parent chunks) fits easily. This matters beyond latency: the original
# one-call-per-chunk map-reduce was the main source of free-tier rate-limit
# exhaustion, since even a modest PDF fanned out into 8-15+ sequential
# requests (each retried up to 3x on Groq before falling back to NVIDIA).
SINGLE_SHOT_CHAR_BUDGET = 24000  # ~6k tokens, comfortably inside gpt-oss-120b's context
BATCH_SIZE = 5                  # parent chunks per map call, when map-reduce is needed
MAX_PARENT_DOCS = 40            # hard cap so a large corpus can't trigger a huge fan-out

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

# Map-reduce fallback for documents over the single-shot budget. Batches
# several parent chunks per map call (rather than one call per chunk) to
# keep the total request count bounded on larger documents.
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

# --- Prompt for source resolution (summary path, multi-document corpora) ---
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
    Classifies the raw query as "qa", "summary", or "chitchat" so graph.py
    can route to the appropriate path. Runs first, before reformulation,
    since reformulation only makes sense for the QA path.
    """
    raw_query = state.get("raw_query", "")
    raw_intent = intent_chain.invoke({"query": raw_query}).strip().lower()

    # Fail safe: an unparseable classification defaults to "qa" rather than
    # silently dropping into a path that might not produce an answer at all.
    intent = raw_intent if raw_intent in _VALID_INTENTS else "qa"

    return {"intent": intent}


def reformulate_query_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Analyzes chat history and reformulates the raw query into an optimized
    search query for the vector database.
    """
    raw_query = state.get("raw_query")
    messages = state.get("messages", [])

    # Short-circuit if there is no chat history (just the current message)
    if len(messages) <= 1:
        return {"search_query": raw_query}

    # Format chat history (excluding the current question at the end)
    chat_history_str = get_buffer_string(messages[:-1])
    current_question = messages[-1].content if hasattr(messages[-1], 'content') else raw_query

    # Invoke the chain
    search_query = rewriter_chain.invoke({
        "chat_history": chat_history_str,
        "current_question": current_question
    })

    # FIX (hallucination safety net): even with the hardened prompt above, a
    # small model can still occasionally answer instead of rewriting. A
    # genuine rewritten search query is almost always short (a few words to
    # one short sentence); an *answer* is typically several sentences long.
    # If the rewrite blows past a generous length bound, treat it as a
    # failed rewrite and fall back to the original question rather than
    # searching with a hallucinated query.
    MAX_REWRITE_WORDS = 25
    if len(search_query.split()) > MAX_REWRITE_WORDS:
        search_query = current_question

    return {"search_query": search_query}


def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Takes the optimized search query and retrieves relevant document
    chunks (expanded to their parent context) using the ensemble retriever.
    """
    # Extract the search_query from the state
    search_query = state.get("search_query")

    # Safety check
    if not search_query:
        search_query = state.get("raw_query")

    ensemble_retriever = _get_ensemble_retriever()
    results = ensemble_retriever.invoke(search_query)

    # NEW: if the user pinned a specific document via the sidebar, drop
    # anything from other sources. Retrieving un-scoped first and filtering
    # after is simpler and more robust than threading a filter through both
    # legs of the ensemble (BM25 + Chroma), and costs nothing extra since
    # k=10 already over-fetches.
    target_source = state.get("target_source")
    if target_source:
        results = [d for d in results if d.metadata.get("source") == target_source]

    return {"documents": results}


def grade_documents_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Filters retrieved documents using a local FlashRank cross-encoder.
    Drops any documents below the dynamic relevance threshold.
    """
    documents = state.get("documents", [])
    search_query = state.get("search_query", "")

    # FIX: threshold now comes from LangGraph's native `config["configurable"]`
    # namespace (passed via app.invoke(inputs, config=...)) instead of being
    # smuggled through the state dict as a plain field.
    configurable = (config or {}).get("configurable", {})
    threshold = configurable.get("relevance_threshold", 0.2)

    # Short-circuit if no documents were retrieved at all
    if not documents:
        return {"documents": []}

    # Compress/Grade the documents
    compressed_docs = compressor.compress_documents(documents, search_query)

    # Filter docs using a list comprehension
    filtered_docs = [
        doc for doc in compressed_docs
        if doc.metadata.get("relevance_score", 0.0) >= threshold
    ]

    return {"documents": filtered_docs}


def generate_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Generates an answer based on the retrieved and filtered documents.
    """
    # Extracting documents and raw query
    documents = state.get("documents", [])
    raw_query = state.get("raw_query")

    # Formatting the context
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
    NEW: summary path. Bypasses similarity search entirely — resolves which
    ingested document the user means, pulls every parent chunk for that
    source straight from the docstore, and summarizes it with the big-tier
    cloud LLM (single call when it fits, batched map-reduce otherwise).
    """
    raw_query = state.get("raw_query", "")
    sources = _list_available_sources()

    if not sources:
        return {"generation": "No documents have been ingested yet."}

    # NEW: if the user pinned a document via the sidebar scope selector,
    # use it directly and skip the LLM-based guess entirely.
    target_source = state.get("target_source")
    if not target_source:
        # Skip the source-resolution LLM call when there's only one
        # candidate document — the common case for a single-resume project.
        if len(sources) == 1:
            target_source = sources[0]
        else:
            target_source = source_match_chain.invoke({
                "sources": "\n".join(sources),
                "query": raw_query,
            }).strip()
            # Fail safe: if the model returns something outside the known
            # list, fall back to the first source rather than failing.
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
    NEW: chitchat path. Skips retrieval entirely for greetings/small talk —
    routing these into generate_node would hit the "I cannot find the
    answer in the provided documents" grounding fallback, since there's no
    retrieved context, which reads as broken rather than just unhelpful.
    """
    response = chitchat_chain.invoke({"query": state.get("raw_query", "")})
    return {"generation": response}