import re
from typing import List, Dict, Tuple, Any
import config

class ParentChildSplitter:
    """
    Implements Parent-Child Document Retrieval logic.
    Splits text into large 'Parent' blocks for context, 
    and small 'Child' blocks for precise vector search.
    """
    
    def __init__(
        self, 
        parent_size: int = 1200, 
        child_size: int = 400, 
        parent_overlap: int = 100,
        child_overlap: int = 50
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Takes raw text and returns a list of Child chunks, 
        each linked to its Parent text in metadata.
        """
        # 1. Create Parent Chunks
        parents = self._sliding_window(text, self.parent_size, self.parent_overlap)
        
        chunks_with_metadata = []
        global_child_index = 0

        for p_index, parent_text in enumerate(parents):
            # 2. Create Child Chunks derived from this Parent
            children = self._sliding_window(parent_text, self.child_size, self.child_overlap)
            
            for c_text in children:
                # 3. Link Child to Parent
                chunk_data = {
                    "text": c_text,  # This is what gets Embedded (Search Key)
                    "metadata": {
                        "parent_text": parent_text,  # This is what gets sent to LLM (Context)
                        "parent_id": f"parent_{p_index}",
                        "is_parent_child": True,
                        "child_index": global_child_index
                    }
                }
                chunks_with_metadata.append(chunk_data)
                global_child_index += 1
                
        return chunks_with_metadata

    def _sliding_window(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Standard sliding window generator"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only keep valid-looking chunks (remove tiny noise)
            if len(chunk.strip()) > 20:
                chunks.append(chunk)
            
            # Prevent infinite loop if overlap >= chunk_size (sanity check)
            step = max(1, chunk_size - overlap)
            start += step
            
        return chunks

# Integration Helper
def process_pdf_parent_child(pdf_file: Any) -> Tuple[List[str], List[Dict], Dict]:
    """
    Pipeline: Extract -> Clean -> Parent-Child Split
    Returns: (Child_Texts, Metadatas, Stats)
    """
    # Import here to avoid circular dependency
    from src.utils import extract_text_from_pdf, clean_text
    
    # 1. Extract & Clean
    raw_text = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_text(raw_text)
    
    # 2. Advanced Splitting
    splitter = ParentChildSplitter(
        parent_size=1000,  # Big context window
        child_size=300,    # Precise search window
        parent_overlap=100,
        child_overlap=50
    )
    
    structured_chunks = splitter.split_text(cleaned_text)
    
    # 3. Unpack for ChromaDB
    child_texts = [c['text'] for c in structured_chunks]
    metadatas = [c['metadata'] for c in structured_chunks]
    
    stats = {
        "num_chunks": len(child_texts),
        "num_parents": len(set(m['parent_id'] for m in metadatas)),
        "cleaned_chars": len(cleaned_text),
        "strategy": "Parent-Child"
    }
    
    return child_texts, metadatas, stats