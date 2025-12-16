import fitz  # PyMuPDF
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import config

def extract_text_from_pdf(pdf_file: Any) -> str:
    """Extract raw text using PyMuPDF."""
    # Reset file pointer just in case
    pdf_file.seek(0)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text: str) -> str:
    """Remove noise (page nums, urls, emails, short lines)."""
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove page numbers (isolated digits)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Sliding window chunking."""
    if not text:
        return []
        
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add chunk if it has meaningful content
        if len(chunk.strip()) > 50: 
            chunks.append(chunk)
            
        start += (chunk_size - overlap)
        
    return chunks

def process_pdf(pdf_file: Any) -> Tuple[List[str], Dict[str, Any]]:
    """
    Pipeline: Extract -> Clean -> Chunk.
    Returns chunks and stats dict for app.py.
    """
    raw_text = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    
    # Stats expected by app.py line 195
    stats = {
        "num_chunks": len(chunks),
        "cleaned_chars": len(cleaned_text),
        "raw_chars": len(raw_text)
    }
    return chunks, stats

def format_sources(sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group source chunks by PDF filename for the UI."""
    grouped = defaultdict(list)
    for src in sources:
        name = src.get('pdf_name', 'Unknown')
        grouped[name].append(src)
    return dict(grouped)

def get_confidence_badge(confidence: float) -> Tuple[str, str]:
    """
    Return emoji and css class based on score.
    Input confidence is expected to be 0-100 scale here (adjusted in retrieval.py).
    """
    if confidence >= 60:
        return "🟢", "confidence-high"
    elif confidence >= 45:
        return "🟡", "confidence-medium"
    else:
        return "🔴", "confidence-low"