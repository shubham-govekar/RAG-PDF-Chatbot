import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyMuPDFLoader

def run_ingestion():
    print("Starting Offline Ingestion Pipeline")
    data_folder = Path("data")
    
    # 1. Defining Splitters
    print("Setting up text splitters...")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # 2. Defining Stores
    print("Initializing ChromaDB and LocalFileStore")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vector_store = Chroma(
        collection_name="rag_data", 
        embedding_function=embeddings, 
        persist_directory="./chroma_db_data"
    )
    
    # Use a relative path so it works on any OS
    fs = LocalFileStore("./parent_docs")
    docstore = create_kv_docstore(fs)

    # 3. Defining the Orchestrator
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        docstore=docstore
    )

    # 4. Loading Documents
    docs = []
    print(f"Scanning {data_folder} for PDFs...")
    
    for pdf_path in data_folder.glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    if not docs:
        print("No PDFs found in the 'data' folder. Exiting.")
        return

    # 5. Ingesting
    print(f"Processing {len(docs)} pages... (This might take a minute)")
    retriever.add_documents(docs)
    print("Ingestion Complete! Data is saved to disk. You can now close this script.")

if __name__ == "__main__":
    # Ensure the data folder exists before trying to read from it
    os.makedirs("data", exist_ok=True)
    run_ingestion()