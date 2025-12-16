import streamlit as st
import base64

# ============================================================
# 🎨 COLOR PALETTE & MODERN GLOSSY STYLES
# ============================================================
def load_custom_css():
    st.markdown("""
    <style>
        /* 1. Main Background & Font */
        .stApp {
            background-color: #0E1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* 2. Chat Message Bubbles */
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* User Bubble (Right/Blue-ish) */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* AI Bubble (Left/Transparent) */
        div[data-testid="stChatMessage"]:nth-child(even) {
            background: transparent;
        }

        /* 3. The "Reasoning Trace" Expander */
        .streamlit-expanderHeader {
            background-color: #1E2329 !important;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #A0A0A0;
        }
        
        /* 4. Sidebar "Glass" Look */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* 5. Custom Buttons (Gradient) */
        .stButton button {
            background: linear-gradient(45deg, #2b5876, #4e4376);
            color: white;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        /* 6. Source Badges */
        .confidence-high { color: #00FF94; font-weight: bold; }
        .confidence-med { color: #FFD700; font-weight: bold; }
        .confidence-low { color: #FF4B4B; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 🧩 UI COMPONENTS
# ============================================================
def display_header(title="Research Assistant AI", subtitle="Agentic RAG • Hybrid Search • 100% Local"):
    """
    Displays the application header with a glassmorphism effect.
    Accepts title and subtitle arguments to match app.py calls.
    """
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {title}
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """
    Displays a professional footer with tech stack badges.
    """
    st.markdown("""
    <style>
        .footer {
            width: 100%;
            margin-top: 50px;
            padding: 20px 0;
            text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #8b949e;
            font-size: 0.9rem;
        }
        .footer a {
            color: #58a6ff;
            text-decoration: none;
            transition: color 0.3s;
        }
        .footer a:hover {
            color: #79c0ff;
            text-decoration: underline;
        }
        .tech-badge {
            display: inline-block;
            padding: 2px 8px;
            margin: 0 4px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.8rem;
            color: #c9d1d9;
        }
    </style>
    
    <div class="footer">
        <p>
            Built with 
            <span class="tech-badge">🦙 Llama 3.2</span>
            <span class="tech-badge">⚡ FlashRank</span>
            <span class="tech-badge">🔍 ChromaDB</span>
        </p>
        <p style="margin-top: 10px;">
            Designed for <b>High-Precision RAG</b> on Local Hardware.
        </p>
    </div>
    """, unsafe_allow_html=True)