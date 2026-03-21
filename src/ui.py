import streamlit as st

# ==============================================================================
# 1. STYLE CONSTANTS
# ==============================================================================

GLOSSY_CSS = """
<style>
    /* GLOBAL FONTS & BACKGROUND */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }

    /* --------------------------------------------------------- */
    /* SIDEBAR STYLES */
    /* --------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2B2F36;
    }

    /* "New Chat" Button */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #1A1D21;
        color: #E3E3E3;
        border: 1px solid #363B42;
        border-radius: 20px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #2B2F36;
        border-color: #5F6368;
        color: #FFF;
    }

    /* --------------------------------------------------------- */
    /* 5. HISTORY ITEMS (THE "UNIFIED PILL" FIX) */
    /* --------------------------------------------------------- */
    
    /* Target the Horizontal Block (The container of the two columns) */
    /* We turn THIS into the button background */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        background-color: transparent; 
        border-radius: 8px;
        gap: 0rem !important; /* REMOVE GAP */
        padding: 0px;
        align-items: center;
        transition: background-color 0.2s;
        margin-bottom: 4px;
    }

    /* Hover effect on the whole row */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        background-color: #1A1D21;
    }

    /* Make the actual buttons transparent so the row background shows */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #9AA0A6 !important;
        text-align: left !important;
        padding: 8px 10px !important;
        margin: 0 !important;
        height: auto !important;
        min-height: 0px !important;
    }

    /* Specific Styles for Text (Left Button) */
    [data-testid="stSidebar"] [data-testid="column"]:first-child button {
        width: 100%;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* Specific Styles for 'X' (Right Button) */
    [data-testid="stSidebar"] [data-testid="column"]:last-child button {
        width: 100%;
        text-align: center !important;
        justify-content: center !important;
        color: #5F6368 !important; 
        padding-right: 8px !important;
    }
    
    /* Hover on X should turn red */
    [data-testid="stSidebar"] [data-testid="column"]:last-child button:hover {
        color: #F28B82 !important;
        background-color: transparent !important;
    }

    /* --------------------------------------------------------- */
    /* CHAT BUBBLES */
    /* --------------------------------------------------------- */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: transparent !important;
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #16181C !important;
        border-radius: 12px;
        padding-right: 20px;
    }
    .stChatMessage .stAvatar {
        background-color: transparent !important;
    }
    
    /* Input Field */
    .stChatInput textarea {
        background-color: #1A1D21 !important;
        color: #E3E3E3 !important;
        border: 1px solid #363B42 !important;
        border-radius: 12px !important;
    }
    
    /* Badges */
    .confidence-high { color: #81C995; font-weight: 500; font-size: 0.8rem; }
    .confidence-med { color: #FDD663; font-weight: 500; font-size: 0.8rem; }
    .confidence-low { color: #F28B82; font-weight: 500; font-size: 0.8rem; }
</style>
"""

# HTML STRING - IMPLICIT CONCATENATION (Safe)
# HTML STRING - IMPLICIT CONCATENATION (Safe)
HERO_HTML = (
    '<div style="text-align: center; margin-top: 80px;">'
    '<h2 style="font-size: 2.2rem; font-weight: 600; color: #E8EAED; margin-bottom: 10px;">What are we analyzing?</h2>'
    '<p style="color: #9AA0A6; font-size: 1.1rem; margin-bottom: 40px;">Open <b>Manage Documents</b> in the sidebar to upload PDFs and begin.</p>'
    '<div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">'
    '<div style="background: #16181C; padding: 20px; border-radius: 12px; width: 200px; text-align: left; border: 1px solid #2B2F36;">'
    '<div style="font-size: 1.5rem; margin-bottom: 10px;">📚</div>'
    '<div style="font-weight: 500; color: #E3E3E3; margin-bottom: 5px;">Context Aware</div>'
    '<div style="font-size: 0.85rem; color: #9AA0A6;">Full document understanding.</div>'
    '</div>'
    '<div style="background: #16181C; padding: 20px; border-radius: 12px; width: 200px; text-align: left; border: 1px solid #2B2F36;">'
    '<div style="font-size: 1.5rem; margin-bottom: 10px;">🔍</div>'
    '<div style="font-weight: 500; color: #E3E3E3; margin-bottom: 5px;">Hybrid Search</div>'
    '<div style="font-size: 0.85rem; color: #9AA0A6;">Keywords + Vectors.</div>'
    '</div>'
    '<div style="background: #16181C; padding: 20px; border-radius: 12px; width: 200px; text-align: left; border: 1px solid #2B2F36;">'
    '<div style="font-size: 1.5rem; margin-bottom: 10px;">🛡️</div>'
    '<div style="font-weight: 500; color: #E3E3E3; margin-bottom: 5px;">Private</div>'
    '<div style="font-size: 0.85rem; color: #9AA0A6;">100% Local Execution.</div>'
    '</div>'
    '</div>'
    '</div>'
)

# ==============================================================================
# 2. RENDER FUNCTIONS
# ==============================================================================

def load_custom_css():
    st.markdown(GLOSSY_CSS, unsafe_allow_html=True)

def display_header(title="Research Assistant", subtitle="Workspace"):
    html = f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #E3E3E3; font-weight: 600; font-size: 1.5rem;">{title}</h1>
        <p style="color: #9AA0A6; font-size: 0.95rem;">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_hero():
    st.markdown(HERO_HTML, unsafe_allow_html=True)

def render_source_card(pdf_name, text, score):
    # Badge Logic
    if score >= 0.7: badge = f'<span class="confidence-high">High ({score:.2f})</span>'
    elif score >= 0.4: badge = f'<span class="confidence-med">Med ({score:.2f})</span>'
    else: badge = f'<span class="confidence-low">Low ({score:.2f})</span>'
    
    st.markdown(f"""
    <div style="padding: 12px; border: 1px solid #2B2F36; border-radius: 8px; margin-bottom: 10px; background: #16181C;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: #E8EAED; font-size: 0.85rem; font-weight: 500;">📄 {pdf_name}</span>
            {badge}
        </div>
        <div style="color: #9AA0A6; font-size: 0.9rem; line-height: 1.5;">
            "{text[:200]}..."
        </div>
    </div>
    """, unsafe_allow_html=True)