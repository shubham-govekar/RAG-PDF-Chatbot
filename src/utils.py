import pymupdf4llm
import re
import os
import tempfile
from typing import Any

def extract_text_as_markdown(pdf_file: Any) -> str:
    """
    Extracts text as Markdown using pymupdf4llm.
    Preserves Tables, Headers, and Equations.
    """
    try:
        # Create a temporary file because pymupdf4llm expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        # Convert to Markdown (Preserves tables as | Col | Col |)
        md_text = pymupdf4llm.to_markdown(tmp_path)
        
        # Cleanup
        os.remove(tmp_path)
        return md_text
        
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def clean_markdown(text: str) -> str:
    """
    Markdown-safe cleaning. 
    Unlike plain text cleaning, we MUST preserve newlines and pipe characters for tables.
    """
    # 1. Remove page numbers (isolated digits on their own line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Fix common markdown issues (e.g. ###Header -> ### Header)
    text = re.sub(r'^(#+)([^#\s])', r'\1 \2', text, flags=re.MULTILINE)
    
    # 3. Collapse excessive vertical whitespace (3+ newlines -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip() 