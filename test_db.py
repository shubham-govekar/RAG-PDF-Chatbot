import chromadb
import os
import config

def diagnose_chroma():
    print("🔍 Starting ChromaDB Diagnostic...")
    
    # Locate the database
    db_path = os.path.join(os.getcwd(), "chroma_db_data")
    if not os.path.exists(db_path):
        print("❌ Error: ChromaDB folder not found at", db_path)
        return

    try:
        # Connect to the local client
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(config.COLLECTION_NAME)
        
        # Get basic stats
        doc_count = collection.count()
        print(f"✅ Collection '{config.COLLECTION_NAME}' loaded successfully.")
        print(f"📊 Total chunks in database: {doc_count}")
        
        # Extract unique PDF names
        if doc_count > 0:
            all_docs = collection.get(include=['metadatas'])
            unique_pdfs = set([meta.get('pdf_name', 'Unknown') for meta in all_docs['metadatas']])
            
            print(f"\n📄 Unique PDFs currently stored ({len(unique_pdfs)}):")
            for pdf in unique_pdfs:
                print(f"   - {pdf}")
        else:
            print("⚠️ Database is currently empty.")
            
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")

if __name__ == "__main__":
    diagnose_chroma()