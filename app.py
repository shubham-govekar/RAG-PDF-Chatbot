import streamlit as st
import chromadb
import os
import uuid
import traceback
import time
import config

# Internal Modules
from src.embeddings import get_embedding_service
from src.generation import get_generation_service
from src.advanced_chunking import process_pdf_parent_child
import src.ui as ui 
from src.session_manager import get_all_sessions, load_session, save_session, delete_session

# Import Retrieval Logic
if config.USE_HYBRID_SEARCH:
    from src.hybrid_retrieval import get_hybrid_retrieval_service as get_retrieval_service
    from src.hybrid_retrieval import reformulate_query, expand_query
else:
    from src.retrieval import get_retrieval_service, reformulate_query

# ============================================================
# 1. SETUP
# ============================================================
st.set_page_config(
    page_title="Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
ui.load_custom_css()

# Database
DB_PATH = os.path.join(os.getcwd(), "chroma_db_data")
client = chromadb.PersistentClient(path=DB_PATH)

# Initialize Collection Early for Sidebar UI
if 'collection' not in st.session_state:
    try: 
        st.session_state.collection = client.get_collection(config.COLLECTION_NAME)
    except Exception: 
        st.session_state.collection = None

# Session
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'conversation_context' not in st.session_state: st.session_state.conversation_context = ""

# Services
embedding_service = get_embedding_service()
retrieval_service = get_retrieval_service()
generation_service = get_generation_service()

# --- MISSING FIX RESTORED: Rebuild BM25 Index on Startup ---
if config.USE_HYBRID_SEARCH and st.session_state.collection:
    search_strategy = retrieval_service.strategies.get('SEARCH')
    if search_strategy and not search_strategy.bm25:
        all_docs = st.session_state.collection.get(include=['documents', 'metadatas'])
        if all_docs and all_docs.get('documents'):
            retrieval_service.build_bm25_index(all_docs['documents'], all_docs['metadatas'])
# -----------------------------------------------------------

# ============================================================
# 2. SIDEBAR (The Command Center)
# ============================================================
with st.sidebar:
    # 1. Primary Action
    if st.button("＋ New chat", use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.conversation_context = ""
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    
    # 2. Context & Knowledge Management
    st.caption("🧠 KNOWLEDGE BASE")
    
    active_pdf = None
    unique_pdfs = []
    
    if st.session_state.get('collection'):
        all_docs = st.session_state.collection.get(include=['metadatas'])
        if all_docs and all_docs.get('metadatas'):
            unique_pdfs = list(set([meta['pdf_name'] for meta in all_docs['metadatas']]))
            
    if unique_pdfs:
        # Active Document Selector
        unique_pdfs_dropdown = ["All Documents"] + unique_pdfs
        selected = st.selectbox("🎯 Active Context", unique_pdfs_dropdown, label_visibility="collapsed")
        if selected != "All Documents":
            active_pdf = [selected]
            
        # Knowledge Manager Expander
        with st.expander("📂 Manage Documents", expanded=False):
            # Uploader moved inside the sidebar
            uploaded_files = st.file_uploader("Add new PDFs", type="pdf", accept_multiple_files=True, key="uploader_populated")
            if uploaded_files and st.button("Ingest", key="ingest_populated"):
                with st.spinner("Ingesting and embedding chunks..."):
                    embedding_service.load_model()
                    for pdf in uploaded_files:
                        child_texts, metadatas, stats = process_pdf_parent_child(pdf)
                        embeddings = embedding_service.embed_texts(child_texts)
                        for i, meta in enumerate(metadatas):
                            meta["source"] = pdf.name
                            meta["pdf_name"] = pdf.name
                            if "chunk_index" not in meta:
                                meta["chunk_index"] = meta.get("child_index", i)
                        ids = [f"{pdf.name}_child_{i}_{uuid.uuid4().hex[:4]}" for i in range(len(child_texts))]
                        st.session_state.collection.add(embeddings=embeddings, documents=child_texts, metadatas=metadatas, ids=ids)
                    
                    if config.USE_HYBRID_SEARCH:
                        docs = st.session_state.collection.get()
                        if docs['documents']: retrieval_service.build_bm25_index(docs['documents'], docs['metadatas'])
                    st.rerun()
                    
            st.divider()
            st.write("Stored Files:")
            
            # The "Delete Specific File" Logic
            for pdf in unique_pdfs:
                col1, col2 = st.columns([0.85, 0.15])
                col1.markdown(f"<span style='font-size: 0.8rem;'>📄 {pdf[:20]}</span>", unsafe_allow_html=True)
                if col2.button("✕", key=f"del_pdf_{pdf}", help=f"Remove {pdf} from database"):
                    st.session_state.collection.delete(where={"pdf_name": pdf})
                    st.rerun()
    else:
        # Empty state UI
        st.info("Database is empty. Upload a PDF to begin.")
        uploaded_files = st.file_uploader("Add new PDFs", type="pdf", accept_multiple_files=True, key="uploader_empty")
        if uploaded_files and st.button("Ingest", key="ingest_empty"):
            with st.spinner("Ingesting and embedding chunks..."):
                embedding_service.load_model()
                collection = client.get_or_create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space": config.DISTANCE_METRIC})
                st.session_state.collection = collection
                
                for pdf in uploaded_files:
                    child_texts, metadatas, stats = process_pdf_parent_child(pdf)
                    embeddings = embedding_service.embed_texts(child_texts)
                    for i, meta in enumerate(metadatas):
                        meta["source"] = pdf.name
                        meta["pdf_name"] = pdf.name
                        if "chunk_index" not in meta:
                            meta["chunk_index"] = meta.get("child_index", i)
                    ids = [f"{pdf.name}_child_{i}_{uuid.uuid4().hex[:4]}" for i in range(len(child_texts))]
                    collection.add(embeddings=embeddings, documents=child_texts, metadatas=metadatas, ids=ids)
                
                if config.USE_HYBRID_SEARCH:
                    docs = collection.get()
                    if docs['documents']: retrieval_service.build_bm25_index(docs['documents'], docs['metadatas'])
                st.rerun()

    st.markdown("---")
    
    # 3. Chat History (Moved to bottom)
    st.caption("💬 RECENT CHATS")
    recent_sessions = get_all_sessions()
    for session in recent_sessions:
        title = session['title']
        if len(title) > 30: title = title[:27] + "..."
        
        col_chat, col_del = st.columns([0.85, 0.15])
        
        with col_chat:
            is_active = session["id"] == st.session_state.session_id
            label = f"🔵 {title}" if is_active else title
            
            if st.button(label, key=f"load_{session['id']}"):
                data = load_session(session['id'])
                if data:
                    st.session_state.chat_history = data['messages']
                    st.session_state.conversation_context = data.get('context', "")
                    st.session_state.session_id = session['id']
                    st.rerun()
        
        with col_del:
            if st.button("✕", key=f"del_{session['id']}", help="Delete"):
                delete_session(session['id'])
                if session['id'] == st.session_state.session_id:
                     st.session_state.chat_history = []
                     st.session_state.session_id = str(uuid.uuid4())
                st.rerun()

# ============================================================
# 3. MAIN WORKSPACE
# ============================================================
ui.display_header(subtitle=config.OLLAMA_MODEL)

# --- Hero ---
if not st.session_state.chat_history:
    ui.display_hero()

# --- Chat Interface (With Avatars) ---
for msg in st.session_state.chat_history:
    avatar_icon = "🧑‍💻" if msg['role'] == "user" else "🤖"
    
    with st.chat_message(msg['role'], avatar=avatar_icon):
        st.markdown(msg['content'])
        
        if msg['role'] == 'assistant' and 'sources' in msg and msg['sources']:
            with st.expander("Verified Sources", expanded=False):
                for src in msg['sources']:
                    ui.render_source_card(
                        pdf_name=src.get('pdf_name', 'Unknown'),
                        text=src.get('text', ''),
                        score=src.get('rerank_score', src.get('score', 0))
                    )
        
        # Display the performance stats badge
        if msg['role'] == 'assistant' and 'stats' in msg and msg['stats']:
            st.caption(f"⚡ {msg['stats']}")

# --- Input ---
# Disable chat input if database is empty
chat_disabled = not bool(unique_pdfs)
placeholder_text = "Upload a PDF in the sidebar to begin..." if chat_disabled else "Ask about your documents..."

if question := st.chat_input(placeholder_text, disabled=chat_disabled):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"): 
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        
        try:
            start_time = time.time()  # Start the performance timer
            context_chunks = []
            final_scores = []
            stats_msg = ""
            expanded_msg = ""
            
            if st.session_state.collection:
                reformulated = reformulate_query(question, st.session_state.conversation_context)
                current_intent = generation_service.detect_intent(reformulated)
                
                search_text = reformulated
                if current_intent == 'SEARCH' and config.ENABLE_QUERY_EXPANSION:
                    expanded_terms = expand_query(reformulated)
                    if expanded_terms:
                        search_text = f"{reformulated} {expanded_terms}"
                        expanded_msg = " | Expanded Query"
                
                results = retrieval_service.retrieve(
                    intent=current_intent,
                    collection=st.session_state.collection,
                    query_text=search_text,  
                    query_emb=embedding_service.embed_query(reformulated),
                    filter_pdfs=active_pdf 
                )
                
                if results['scores']:
                    top = results['scores'][0].get('rerank_score', results['scores'][0].get('confidence', 0))
                    if top >= 0.2:
                        context_chunks = [s.get('parent_text', s['text']) for s in results['scores']]
                        final_scores = results['scores']

            full_response = ""
            stream = generation_service.generate_answer_stream(question, context_chunks, st.session_state.chat_history)
            
            for chunk in stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            
            # Calculate Latency and build the stats string
            latency = round(time.time() - start_time, 1)
            if final_scores:
                stats_msg = f"Searched {len(final_scores)} chunks{expanded_msg} | {latency}s"
                st.caption(f"⚡ {stats_msg}")
            
            st.session_state.conversation_context = f"User: {question}\nAI: {full_response}"
            st.session_state.chat_history.append({
                'role': 'assistant', 
                'content': full_response, 
                'sources': final_scores,
                'stats': stats_msg
            })
            save_session(st.session_state.session_id, st.session_state.chat_history, st.session_state.conversation_context)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())