"""
RAG Chatbot Phase 3: Parent-Child Retrieval + Hybrid Search + Modern UI
"""
import streamlit as st
import chromadb
from datetime import datetime
import traceback

# Import internal modules
import config
from src.embeddings import get_embedding_service
from src.generation import get_generation_service
from src.utils import format_sources, get_confidence_badge
from src.advanced_chunking import process_pdf_parent_child
import src.ui as ui  # Your new Glossy UI module

# Import appropriate retrieval service based on config
if config.USE_HYBRID_SEARCH:
    from src.hybrid_retrieval import get_hybrid_retrieval_service as get_retrieval_service
    from src.hybrid_retrieval import reformulate_query
else:
    from src.retrieval import get_retrieval_service, reformulate_query

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="centered"
)

# 1. LOAD STYLES & HEADER
ui.load_custom_css()

phase_info = "Phase 3: Hybrid + Parent-Child" if config.USE_HYBRID_SEARCH else "Phase 1: Vector Search"
ui.display_header(
    title="PDF RAG Assistant", 
    subtitle=f"{phase_info} | Powered by Llama 3.2"
)

# Optional Hero Image
if 'processed' not in st.session_state:
    st.session_state.processed = False
    

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'pdf_names' not in st.session_state:
    st.session_state.pdf_names = []
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = ""
if 'auto_question' not in st.session_state:
    st.session_state.auto_question = ''
    # Initialize a counter to force-reset the uploader
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Initialize services
embedding_service = get_embedding_service()
retrieval_service = get_retrieval_service()
generation_service = get_generation_service()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📄 Upload PDFs")
    
    # Reset Button
    if st.session_state.processed:
        if st.button("🔄 Reset & Upload New"):
            # Clear internal processing state
            for key in ['processed', 'collection', 'pdf_names', 'chunk_count', 
                        'chat_history', 'conversation_context', 'selected_pdfs']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # THE FIX: Increment key to force a brand new uploader widget
            st.session_state.uploader_key += 1 
            st.rerun()
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        # THE FIX: Use the dynamic key
        key=f"uploader_{st.session_state.uploader_key}" 
    )
    
    # PDF Filter (Focus Mode)
    if st.session_state.processed and len(st.session_state.pdf_names) > 1:
        st.markdown("---")
        st.subheader("🔍 Focus Mode")
        selected_pdfs = st.multiselect(
            "Search only in:",
            options=st.session_state.pdf_names,
            default=st.session_state.pdf_names
        )
        st.session_state.selected_pdfs = selected_pdfs
    
    # Statistics
    if st.session_state.processed:
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("PDFs Loaded", len(st.session_state.pdf_names))
        st.metric("Total Chunks", st.session_state.chunk_count)
    
    # System Status
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    ollama_status = generation_service.check_ollama_available()
    if ollama_status['available']:
        st.success(f"✅ Ollama: {config.OLLAMA_MODEL}")
    else:
        st.error("❌ Ollama not running")
    
    emb_info = embedding_service.get_model_info()
    if emb_info['loaded']:
        st.success(f"✅ Embeddings: {config.EMBEDDING_MODEL}")
    
    if config.USE_HYBRID_SEARCH:
        st.success(f"✅ Hybrid Search Active")
    
    if config.ENABLE_RERANKING:
        st.success(f"✅ Re-ranker Active")

    # Export Chat
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("💾 Download Chat"):
            export = f"# PDF RAG Chat\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for msg in st.session_state.chat_history:
                role = "User" if msg['role'] == "user" else "AI"
                export += f"**{role}:** {msg['content']}\n\n"
            st.download_button("📥 Save .md", export, file_name="chat.md")

# ============================================================
# PDF PROCESSING (PHASE 3: PARENT-CHILD + ROBUST STATE)
# ============================================================
if uploaded_files and not st.session_state.processed:
    with st.spinner("🔄 Processing PDFs with Advanced Strategy..."):
        try:
            embedding_service.load_model()
            
            # ChromaDB Setup
            client = chromadb.Client()
            try:
                client.delete_collection(config.COLLECTION_NAME)
            except:
                pass
            
            collection = client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": config.DISTANCE_METRIC}
            )
            
            # --- 1. SETUP LOCAL LISTS (Fixes Duplicate Bug) ---
            # We use a local list 'current_pdf_names' instead of appending directly to session state.
            # This ensures we start with 0 names every time we process.
            current_pdf_names = [] 
            all_chunks = []
            all_metadatas = []
            all_ids = []
            chunk_counter = 0
            
            # Process Loop
            for pdf_file in uploaded_files:
                with st.spinner(f"📄 Analyzing {pdf_file.name}..."):
                    # Add name to local list
                    current_pdf_names.append(pdf_file.name)
                    
                    # Phase 3: Parent-Child Processing
                    child_chunks, metadatas, stats = process_pdf_parent_child(pdf_file)
                    
                    st.write(f"✅ **{pdf_file.name}**")
                    st.caption(f"Strategy: {stats['strategy']} | {stats['num_chunks']} children from {stats['num_parents']} parents")
                    
                    # Accumulate data
                    all_chunks.extend(child_chunks)
                    
                    for meta in metadatas:
                        meta['pdf_name'] = pdf_file.name
                        meta['chunk_index'] = meta['child_index']
                        
                        all_metadatas.append(meta)
                        all_ids.append(f"chunk_{chunk_counter}")
                        chunk_counter += 1
            
            # Generate Embeddings
            with st.spinner(f"🧠 Generating vectors for {len(all_chunks)} chunks..."):
                all_embeddings = embedding_service.embed_texts(all_chunks, show_progress=True)
            
            # Store in ChromaDB
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            # --- 2. UPDATE STATE ATOMICALLY (Fixes 0 Chunks Bug) ---
            # Update these BEFORE building BM25, so even if BM25 fails, data is safe.
            st.session_state.pdf_names = current_pdf_names  # Replaces old list completely
            st.session_state.chunk_count = chunk_counter
            st.session_state.collection = collection
            
            # Build BM25 Index
            if config.USE_HYBRID_SEARCH:
                with st.spinner("🔨 Building Keyword Index..."):
                    retrieval_service.build_bm25_index(all_chunks, all_metadatas)
            
            # Finalize
            st.session_state.processed = True
            
            # Force Rerun to update Sidebar immediately
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())

# ============================================================
# CHAT INTERFACE
# ============================================================
if st.session_state.processed:
    # 1. Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            # Show the main answer
            st.markdown(msg['content'])
            
            # Show Sources (Only for Assistant)
            if msg['role'] == 'assistant' and 'sources' in msg:
                with st.expander("📚 View Source Snippets"):
                    for i, source in enumerate(msg['sources']):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.caption(f"**Ref {i+1}**")
                            st.caption(f"📄 {source.get('pdf_name', 'Unknown')}")
                        with col2:
                            raw_text = source.get('text', 'No text available')
                            st.info(raw_text[:400] + "...")

    # 2. Chat Input (MUST BE OUTSIDE THE FOR LOOP)
    question = st.chat_input("Ask about your documents...")
    
    if question:
        # User Message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Assistant Response
        # Assistant Response
        with st.chat_message("assistant"):
            try:
                # --- 1. ANIMATED AGENT WORKFLOW ---
                with st.status("🧠 Agent Reasoning...", expanded=True) as status:
                    
                    # Step A: Reformulation & Routing
                    st.write("🔍 Analyzing query intent...")
                    reformulated = reformulate_query(question, st.session_state.conversation_context)
                    intent = generation_service.detect_intent(reformulated)
                    st.write(f"👉 Detected Intent: **{intent}**")
                    
                    # Step B: Retrieval
                    st.write(f"📚 Searching documents using {intent} strategy...")
                    query_emb = embedding_service.embed_query(reformulated)
                    filter_pdfs = st.session_state.get('selected_pdfs', None)
                    
                    results = retrieval_service.retrieve(
                        intent=intent,
                        collection=st.session_state.collection,
                        query_text=reformulated,
                        query_emb=query_emb,
                        filter_pdfs=filter_pdfs
                    )
                    
                    # --- Step C: Re-ranking & The "Hard Gate" ---
                    if intent == "SEARCH":
                        st.write("✨ Re-ranking results with Cross-Encoder...")
                    
                    # 1. Get the Top Result
                    top_chunk = results['scores'][0] if results['scores'] else None
                    
                    # 2. THE HARD GATE (Crucial Fix)
                    # We look ONLY at 'rerank_score' (The Smart Score).
                    # FlashRank is harsh. A score < 0.2 is almost certainly garbage.
                    # We default to 1.0 if rerank_score is missing (e.g., if intent != SEARCH)
                    raw_score = top_chunk.get('rerank_score', 0.0) if top_chunk else 0
                    
                    # Strict threshold: If the smart model says it's < 20% relevant, kill it.
                    # This works for Sudan, India, France, Mars, etc.
                    is_relevant = top_chunk and raw_score >= 0.2
                    
                    # Update UI based on this robust check
                    if is_relevant:
                        status.update(label=f"✅ Context Retrieved! (Relevance: {raw_score:.2f})", state="complete", expanded=False)
                    else:
                        status.update(label="❌ Query unrelated to Document", state="error", expanded=False)

                # --- 2. DETAILED TRACE ---
                with st.expander("🧠 View Detailed Logic", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Intent", intent)
                        st.metric("Chunks Found", len(results['scores']))
                        # Show the raw score that made the decision
                        if top_chunk:
                            st.metric("Cross-Encoder Score", f"{raw_score:.4f}")
                    # ... (rest of the mermaid code) ...

                # --- 3. GENERATION ---
                if not is_relevant:
                    # ROBUST FAILURE MESSAGE
                    st.warning("⚠️ The AI determined this question is unrelated to the uploaded PDF.")
                    
                    # Optional: Print what it found so you trust it
                    if top_chunk:
                        with st.expander("See what was rejected (for debugging)"):
                            st.write(f"Best bad match: {top_chunk['text'][:200]}...")
                            st.write(f"Relevance Score: {raw_score:.4f}")

                    response_content = "I couldn't find the answer in the provided documents."
                    final_scores = []
                else:
                    # Scenario: Good Context Found
                    context_chunks = [s['text'] for s in results['scores']]
                    
                    full_response = st.write_stream(
                        generation_service.generate_answer_stream(
                            question,
                            context_chunks,
                            st.session_state.chat_history
                        )
                    )
                    response_content = full_response
                    final_scores = results['scores']
                
                # --- 4. SAVE HISTORY ---
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_content,
                    'sources': final_scores
                })
                st.session_state.conversation_context = question
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
else:
    # Empty State Hint
    st.info("👈 Upload your PDF documents in the sidebar to begin!")

# Footer
ui.display_footer()