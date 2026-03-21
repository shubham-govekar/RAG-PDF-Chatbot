import os
import json
import numpy as np
from datetime import datetime

CHATS_DIR = "chats"

# Ensure chats directory exists
if not os.path.exists(CHATS_DIR):
    os.makedirs(CHATS_DIR)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # 1. Handle Numpy (The fix we just applied)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # 2. FUTURE PROOFING: Handle Sets (converts {a,b} -> [a,b])
        if isinstance(obj, set):
            return list(obj)
            
        # 3. FUTURE PROOFING: Handle Bytes (converts b'abc' -> "abc")
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
            
        return super(NpEncoder, self).default(obj)

def get_all_sessions():
    """Returns a list of available chat sessions sorted by newest first."""
    sessions = []
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHATS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data["id"],
                        "title": data.get("title", "Untitled Chat"),
                        "last_updated": data.get("last_updated", "")
                    })
            except Exception:
                continue
    
    # Sort by last updated (descending)
    sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    return sessions

def load_session(session_id):
    """Loads a specific session by ID."""
    filepath = os.path.join(CHATS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_session(session_id, messages, context=None):
    """Saves the current chat state to disk."""
    if not messages:
        return
    
    # Generate a simple title from the first user message if needed
    title = "New Chat"
    for msg in messages:
        if msg["role"] == "user":
            # Take first 6 words of first user question
            title = " ".join(msg["content"].split()[:6]) + "..."
            break
            
    data = {
        "id": session_id,
        "title": title,
        "last_updated": datetime.now().isoformat(),
        "messages": messages,
        "context": context or ""
    }
    
    filepath = os.path.join(CHATS_DIR, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        # THE FIX: Added cls=NpEncoder to handle float32
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NpEncoder)

def delete_session(session_id):
    """Deletes a session file."""
    filepath = os.path.join(CHATS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)