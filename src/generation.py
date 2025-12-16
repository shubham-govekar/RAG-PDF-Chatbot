"""
LLM Generation Service (Ollama)
Now supports Streaming for 'Typewriter Effect'
"""
import ollama
from typing import List, Dict, Generator
import config
import re

def clean_encoding_artifacts(text: str) -> str:
    """
    Cleans common PDF extraction errors found in scientific papers.
    """
    # Map of "Garbage" -> "Correct Symbol"
    # These mappings are specific to the font encoding in your paper
    replacements = {
        '¼': '=',      # The equals sign is often misread as ¼
        '': '-',      # Minus sign
        'ð': '(',      # Open parenthesis
        'Þ': ')',      # Close parenthesis
        '': '*',      # Multiplication/Convolution
        '': ' ',      # Backspace char -> Space
        '␀': '',       # Null bytes
        'hwn': 'w_n',  # Fix merged variable names
        'yn': 'y_n'
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

class GenerationService:
    def __init__(self):
        self.client = ollama

    def check_ollama_available(self) -> Dict[str, bool]:
        """Check if Ollama is running"""
        try:
            self.client.list()
            return {'available': True}
        except Exception:
            return {'available': False}

    def _build_prompt(self, question: str, chunks: List[str], history: List[Dict[str, str]]) -> str:
        """Construct prompt with context and history"""
        # Limit history to avoid context overflow
        recent_history = history[-config.MAX_HISTORY_MESSAGES:]
        
        # Format history
        hist_txt = ""
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            hist_txt += f"{role}: {msg['content']}\n"
        
        cleaned_chunks = [clean_encoding_artifacts(chunk) for chunk in chunks]
        ctx_txt = "\n---\n".join(cleaned_chunks)
        # --------------------------
        
        return (
            "You are a helpful research assistant. Answer the question using ONLY the provided context chunks.\n"
            "Use the conversation history to understand references like 'it' or 'that'.\n"
            "CRITICAL GUIDELINES:\n"
            "1. If the exact word isn't in the text, look for synonyms.\n"
            "2. EXPLAIN MATH: The context may contain complex equations. Do not just copy them. Explain the *components* and what they represent in plain English.\n"
            f"--- Conversation History ---\n{hist_txt}\n\n"
            f"--- Context ---\n{ctx_txt}\n\n"
            f"--- Question ---\n{question}"
        )
    
    def detect_intent(self, query: str) -> str:
        """
        Agentic Router with 'Fast Path' Regex Caching.
        """
        query_lower = query.lower()
        
        # 1. FAST PATH: Regex Heuristics (Zero Latency)
        patterns = {
            "METADATA": [
                r"who (wrote|created|authored) this", 
                r"author(s)?", 
                r"published (in|by|date)", 
                r"reference(s)?",
                r"citation(s)?"
            ],
            # STRICTER Summary Patterns
            "SUMMARY": [
                r"^summarize",  # Starts with summarize
                r"^summary", 
                r"give me an overview", 
                r"^abstract", 
                r"what is this (paper|document|article) about", # Specific phrasing
                r"explain this (paper|document)"
            ],
            # NEW: Force specific questions to SEARCH
            "SEARCH": [
                r"what (is|are|does) .+",  # Matches "What is HACE...", "What does X mean"
                r"how (does|do) .+",       # Matches "How does the model work"
                r"define .+",
                r"explain .+",             # Matches "Explain HACE"
                r"compare .+"
            ]
        }
        
        for intent, regex_list in patterns.items():
            for pattern in regex_list:
                if re.search(pattern, query_lower):
                    return intent

        # 2. SLOW PATH: LLM Classification (Smart fallback)
        # We simplify the prompt for the 1B model
        prompt = (
            f"Classify this query.\n"
            f"Query: {query}\n\n"
            f"Rules:\n"
            f"- If asking for authors, dates, or title -> METADATA\n"
            f"- If asking for a full summary of the whole PDF -> SUMMARY\n"
            f"- If asking for definitions, specific facts, or details -> SEARCH\n\n"
            f"Category (METADATA, SUMMARY, or SEARCH):"
        )
        
        try:
            resp = self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            intent = resp['message']['content'].strip().upper()
            
            if "METADATA" in intent: return "METADATA"
            if "SUMMARY" in intent: return "SUMMARY"
            return "SEARCH" # Default safe option
            
        except Exception:
            return "SEARCH"

    def generate_answer_stream(self, question: str, context_chunks: List[str], history: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Yields response chunks one token at a time.
        This enables the 'Typewriter Effect' in Streamlit.
        """
        prompt = self._build_prompt(question, context_chunks, history)
        
        try:
            # Create a generator that yields tokens as they arrive
            stream = self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': config.TEMPERATURE, 
                    'top_p': config.TOP_P, 
                    'top_k': config.TOP_K, 
                    'num_predict': 512,
                    'repeat_penalty': 1.15  # Prevents looping/repetition
                },
                stream=True  # <--- THE MAGIC SWITCH
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            yield f"Error generating answer: {str(e)}"

    # Keep this for backward compatibility if needed, but we will switch to stream
    def generate_answer(self, question: str, context_chunks: List[str], history: List[Dict[str, str]]) -> str:
        """Non-streaming version (waits for full response)"""
        prompt = self._build_prompt(question, context_chunks, history)
        try:
            resp = self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': config.TEMPERATURE, 
                    'top_p': config.TOP_P, 
                    'top_k': config.TOP_K, 
                    'num_predict': 512,
                    'repeat_penalty': 1.15
                }
            )
            return resp['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

# Global instance
_generation_service = None

def get_generation_service() -> GenerationService:
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service