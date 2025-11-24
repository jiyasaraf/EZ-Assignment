# main.py
import streamlit as st
import os
import tempfile
import uuid # For generating unique chat IDs
from datetime import datetime # For timestamps

# Import custom modules
from mod import process_document, extract_text_from_pdf, extract_text_from_txt
from rag import DocumentRag
from ask_anything import generate_response_with_gemini, generate_summary_with_gemini
from challenge_me import generate_challenge_questions, evaluate_user_answer
import time # For simulating loading delays for better UX


# --- DocumentRag Initialization (Global) ---
# Initialize DocumentRag only once. It will manage local ChromaDB persistence.
if 'document_rag' not in st.session_state:
    # Use a fixed directory for persistence since we're not using Firebase user IDs
    st.session_state.document_rag = DocumentRag(persist_directory="./local_chroma_db")
    print("DEBUG: DocumentRag initialized for local ChromaDB.")


# --- Streamlit UI Configuration ---
st.set_page_config(
    layout="wide",
    page_title="DocuMind",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# Initialize other session state variables
if 'chats' not in st.session_state:
    st.session_state.chats = [] # List of chat dictionaries from ChromaDB
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None # ID of the currently active chat
if 'current_chat_name' not in st.session_state:
    st.session_state.current_chat_name = "New Chat" # Name of the currently active chat
if 'full_document_text' not in st.session_state:
    st.session_state.full_document_text = "" # Full text of the document for current chat
if 'summary' not in st.session_state:
    st.session_state.summary = "" # Summary of the document for current chat
if 'ask_history' not in st.session_state:
    st.session_state.ask_history = [] # Stores tuples of (question, answer, justification) for current chat
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None # Name of the document for current chat
if 'ask_question_input_value' not in st.session_state:
    st.session_state.ask_question_input_value = ""
# Remove dark_mode from session state as it's no longer toggleable
# if 'dark_mode' not in st.session_state:
#     st.session_state.dark_mode = False
if 'current_panel' not in st.session_state:
    st.session_state.current_panel = 'summary' # Default panel to show
if 'is_new_chat_flow' not in st.session_state:
    st.session_state.is_new_chat_flow = True # Flag to indicate if we are in a fresh new chat session
if 'show_search' not in st.session_state:
    st.session_state.show_search = False


# --- Chat Session Management Functions ---

def fetch_all_chats():
    """Fetches all chat sessions from ChromaDB."""
    chats_list = st.session_state.document_rag.fetch_all_chat_metadata()
    st.session_state.chats = chats_list
    print(f"DEBUG: Fetched {len(chats_list)} chats from ChromaDB metadata collection.")
    
# Function to save/update a chat session in ChromaDB
def save_chat_session(chat_id, chat_name, document_name, full_document_text, summary, ask_history, processed_chunks_count):
    chat_data = {
        "chat_id": chat_id,
        "name": chat_name,
        "timestamp": datetime.now().isoformat(), # Use ISO format string for timestamp
        "document_name": document_name,
        "full_document_text": full_document_text,
        "summary": summary,
        "ask_history": ask_history,
        "processed_document_chunks_count": processed_chunks_count
    }
    st.session_state.document_rag.save_chat_metadata(chat_id, chat_data)
    print(f"DEBUG: Chat session '{chat_name}' ({chat_id}) saved/updated in ChromaDB.")

# Function to update only the ask_history for an existing chat
def update_ask_history(chat_id, ask_history):
    """Updates just the ask_history in the chat metadata."""
    st.session_state.document_rag.update_chat_metadata(chat_id, {"ask_history": ask_history})
    print(f"DEBUG: Updated ask_history for chat {chat_id}.")

# Function to load a specific chat session
def load_chat_session(chat_data):
    # Update session state with the loaded chat's data
    st.session_state.current_chat_id = chat_data["chat_id"]
    st.session_state.current_chat_name = chat_data["name"]
    st.session_state.uploaded_file_name = chat_data["document_name"]
    st.session_state.full_document_text = chat_data["full_document_text"]
    st.session_state.summary = chat_data["summary"]
    st.session_state.ask_history = chat_data["ask_history"]
    
    # Clear challenge mode specific states when loading a new chat
    st.session_state.challenge_questions = []
    st.session_state.user_answers = []
    st.session_state.evaluation_results = []
    
    st.session_state.is_new_chat_flow = False # Not a new chat flow when loading existing
    st.session_state.ask_question_input_value = "" # Clear input on new chat load

    print(f"DEBUG: Loaded chat session: {chat_data['name']} ({chat_data['chat_id']})")
    st.session_state.current_panel = 'summary' # Always go to summary after loading a chat
    st.rerun() # Rerun to update the UI with the loaded chat's data

# Function to delete a specific chat session
def delete_chat_session(chat_id):
    try:
        # Delete associated embeddings and metadata from ChromaDB
        st.session_state.document_rag.delete_documents_by_chat_id(chat_id)
        print(f"DEBUG: Data for chat {chat_id} deleted from ChromaDB.")

        # If the deleted chat was the current one, start a new chat
        if st.session_state.current_chat_id == chat_id:
            start_new_chat()
        else:
            fetch_all_chats() # Refresh chat list if a different chat was deleted
            st.rerun()
    except Exception as e:
        st.error(f"Error deleting chat session: {e}")
        print(f"ERROR: Error deleting chat session {chat_id}: {e}")

# Function to start a new chat session
def start_new_chat():
    new_chat_id = str(uuid.uuid4()) # Generate a unique ID for the new chat
    st.session_state.current_chat_id = new_chat_id
    st.session_state.current_chat_name = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Reset all relevant session state variables for a fresh start
    st.session_state.uploaded_file_name = None
    st.session_state.full_document_text = ""
    st.session_state.summary = ""
    st.session_state.ask_history = []
    st.session_state.challenge_questions = []
    st.session_state.user_answers = []
    st.session_state.evaluation_results = []
    st.session_state.is_new_chat_flow = True # Indicate new chat flow
    st.session_state.ask_question_input_value = "" # Clear input box
    st.session_state.current_panel = 'summary' # Default to summary panel for new chat

    print(f"DEBUG: Started new chat session with ID: {new_chat_id}")
    fetch_all_chats() # Refresh the sidebar chat list to show the new chat (initially empty)
    st.rerun()

# --- Initial Load / Chat Management ---
# This block ensures a chat session is active on first load or after certain actions
if not st.session_state.current_chat_id:
    # On initial load, or if no chat is active, start a new one
    start_new_chat()
else:
    # If a chat is already active (e.g., after a rerun), fetch all chats to update sidebar
    # This ensures the sidebar is always up-to-date with ChromaDB
    fetch_all_chats()


# --- Custom CSS for basic styling and dark mode toggle ---
# This CSS will primarily target custom markdown elements and use Streamlit's theming for native widgets.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        /* Default text color, will be overridden by Streamlit's theme or .dark-mode */
        color: #333333; /* Enforce light mode text color */
    }

    /* Define CSS variables for colors, enforcing light mode */
    :root {
        --background-color: #f0f2f6; /* Light mode app background */
        --text-color: #333333; /* Light mode general text */
        --sidebar-bg: #ffffff; /* Light mode sidebar background */
        --panel-bg: #ffffff; /* Light mode main panel background */
        --border-color: #e0e0e0; /* Light mode borders */
        --input-bg: #f9f9f9; /* Light mode input/textarea background */
        --chat-user-bg: #ADD8E6; /* Light blue user chat bubble */
        --chat-ai-bg: #e0e0e0; /* Light mode AI chat bubble */
        --header-bg: #ffffff; /* Light mode header background */
        --active-chat-bg: #e6f7ff; /* Light mode active chat highlight */
        --delete-icon-color: #ff4d4f; /* Red for delete icon */
    }

    /* Remove dark-mode specific styles */
    /* .dark-mode {
        --background-color: #1a1a2e;
        --text-color: #e0e0e0;
        --sidebar-bg: #0f0f1a;
        --panel-bg: #1a1a2e;
        --border-color: #3a3a5a;
        --input-bg: #2a2a4a;
        --chat-user-bg: #ADD8E6;
        --chat-ai-bg: #3a3a5a;
        --header-bg: #1a1a2e;
        --active-chat-bg: #2a2a4a;
        --delete-icon-color: #ff7875;
    } */

    /* Apply base colors to the Streamlit app container */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Main content area styling */
    /* Use data-testid for more stable targeting */
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap { /* This is a common class for the main content block */
        background-color: var(--panel-bg);
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        margin: 1rem;
        height: calc(100vh - 2rem);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        color: var(--text-color); /* Ensure text color is set for main content */
    }
    /* Remove dark-mode specific shadow */
    /* .dark-mode div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    } */

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
        border-radius: 0 1rem 1rem 0;
        color: var(--text-color); /* Ensure text color is set for sidebar */
    }
    /* Remove dark-mode specific shadow */
    /* .dark-mode div[data-testid="stSidebar"] {
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.2);
    } */

    /* Make sidebar content scrollable and hide the toggle button */
    div[data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        height: 95vh; /* Adjust height to fit viewport */
    }
    /* General button styling (targets Streamlit's native buttons) */
    .stButton > button {
        background-color: #ADD8E6; /* Green */
        color: #333333;
        border-radius: 0.75rem;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        transition: background-color 0.2s, box-shadow 0.2s;
        border: none;
    }
    .stButton > button:hover {
        background-color: #e0e0e0; /* Light grey hover */
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .stButton > button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
        box-shadow: none;
    }
    /* Specific styling for the download button */
    div[data-testid="stDownloadButton"] > button {
        background-color: #ADD8E6; /* Light Blue */
        color: #333333 !important; /* Dark text for readability */
        border: 1px solid #ADD8E6 !important;
        background-color: #ADD8E6 !important; /* Light Blue */
    }

    /* Sidebar navigation buttons specific styling */
    div[data-testid="stSidebarNav"] .stButton > button {
        background-color: transparent; /* Override for sidebar buttons */
        color: var(--text-color); /* Use text color variable */
        text-align: left;
        padding-left: 1rem;
    }
    div[data-testid="stSidebarNav"] .stButton > button:hover {
        background-color: #e0e0e0; /* Light hover */
        color: #333333;
    }
    /* Remove dark-mode specific hover */
    /* .dark-mode div[data-testid="stSidebarNav"] .stButton > button:hover {
        background-color: #2a2a4a;
        color: #e0e0e0;
    } */
    /* Active sidebar button styling (Streamlit's primary button style when selected) */
    div[data-testid="stSidebarNav"] .stButton > button[data-testid="stSidebarNav"] {
        background-color: #ADD8E6;
        color: white;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }


    /* File uploader button styling */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.25rem;
        max-width: 700px;
        margin: 0 auto;
    }
    /* Remove dark inner backgrounds and hide default label */
    div[data-testid="stFileUploader"] label { display: none !important; }
    div[data-testid="stFileUploader"] div { background-color: transparent !important; }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: #f9fafb !important;
        border: 1px dashed #d1d5db !important;
        border-radius: 10px !important;
        padding: 2.25rem 1rem !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #007bff;
        color: white;
        display: block;
        margin: 0.75rem auto 0;
    }
    div[data-testid="stFileUploader"] button:hover { background-color: #0056b3; }

    /* Chat message styling */
    .chat-message {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
        margin-bottom: 10px;
        word-wrap: break-word;
        box-shadow: 0 2px 5px (0,0,0,0.1);
    }
    .user-message {
        background-color: var(--chat-user-bg);
        align-self: flex-end;
        margin-left: auto;
        color: var(--text-color);
    }
    .ai-message {
        background-color: var(--chat-ai-bg);
        align-self: flex-start;
        margin-right: auto;
        color: var(--text-color);
    }

    /* Text area and input styling */
    textarea, input[type="text"] {
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        background-color: var(--input-bg);
        color: var(--text-color);
        padding: 0.75rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #ADD8E6;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
        outline: none;
    }

    /* Info/Success/Error messages */
    div[data-testid="stAlert"] {
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stAlert"].info { background-color: #e0f7fa; color: #00796b; }
    div[data-testid="stAlert"].success { background-color: #e8f5e9; color: #2e7d32; }
    div[data-testid="stAlert"].error { background-color: #ffebee; color: #c62828; }
    div[data-testid="stAlert"].warning { background-color: #fffde7; color: #f9a825; }

    /* Remove dark-mode specific alert styles */
    /* .dark-mode div[data-testid="stAlert"].info { background-color: #004d40; color: #80cbc4; }
    .dark-mode div[data-testid="stAlert"].success { background-color: #1b5e20; color: #a5d6a7; }
    .dark-mode div[data-testid="stAlert"].error { background-color: #b71c1c; color: #ef9a9a; }
    .dark-mode div[data-testid="stAlert"].warning { background-color: #f57f17; color: #ffee58; } */

    /* For the spinner */
    .stSpinner > div > div {
        border-top-color: #ADD8E6;
    }

    /* Adjust padding for the main content area to make it full width */
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-z5fcl4 {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap {
        padding: 2rem;
    }

    /* Fix for the sidebar expand/collapse button text */
    /* Target the button by its data-testid and hide its internal text span */
    button[data-testid="stSidebarToggle"] span {
        visibility: hidden; /* Hide the original text/icon */
        width: 0; /* Collapse its width */
        overflow: hidden; /* Prevent overflow */
    }
    /* Add content for the arrow symbols using ::before or ::after */
    button[data-testid="stSidebarToggle"]::before {
        content: '›'; /* Right arrow for expand (open) */
        font-size: 1.5rem;
        line-height: 1;
        display: inline-block;
        vertical-align: middle;
        color: var(--text-color); /* Ensure arrow color changes with theme */
        visibility: visible; /* Make the pseudo-element visible */
        width: auto;
        padding: 0; /* Remove padding if any */
        margin: 0; /* Remove margin if any */
    }
    /* When sidebar is expanded, change to left arrow for collapse (close) */
    div[data-testid="stSidebar"][aria-expanded="true"] button[data-testid="stSidebarToggle"]::before {
        content: '‹' !important; color: var(--text-color) !important; /* Left arrow for collapse, ensure color is applied */
    }

    /* Hide the sidebar toggle button to make it static */
    button[data-testid="stSidebarToggle"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        overflow-y: auto;
    }
    .chat-menu { padding: 0 0.5rem 0.5rem 0.5rem; }
    .chat-menu .stButton > button { background-color: #eaf4ff; color: #333333; border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 0.25rem 0.5rem; }
    .chat-menu .stButton > button:hover { background-color: #dcecff; }

    /* Style for the copy icon */
    .copy-icon {
        cursor: pointer;
        margin-left: 10px;
        font-size: 1.2em;
        vertical-align: middle;
        color: #e5e7eb;
        transition: color 0.2s;
    }
    .copy-icon:hover { color: #9ca3af; }

    /* Sidebar text-style clickable items */
    .sidebar-brand { font-size: 1rem; font-weight: 700; padding: 0.75rem 0.75rem; color: #ffffff; }
    .sidebar-actions .stButton > button,
    .sidebar-nav .stButton > button {
        background-color: transparent; color: #e5e7eb; text-align: left; border-radius: 0.35rem; padding: 0.4rem 0.6rem; border: 1px solid transparent; font-weight: 500;
    }
    .sidebar-chats .stButton > button {
        background-color: transparent;
        color: #333333;
        border-radius: 0.6rem;
        padding: 0.35rem;
        border: 1px solid var(--border-color);
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-width: 36px;
        min-height: 36px;
    }
    .sidebar-actions .stButton > button:hover,
    .sidebar-nav .stButton > button:hover,
    .sidebar-chats .stButton > button:hover { background-color: #1a1a1a; color: #e5e7eb; }
    .sidebar-section-title { display:flex; align-items:center; gap:0.5rem; padding: 0.5rem 0.75rem; color: #e5e7eb; font-weight:600; }
    .sidebar-section-title .hr { flex:1; height:1px; background:#2a2a2a; }
    .hr-line { height:1px; background:#2a2a2a; margin: 0.5rem 0; }

    /* Chat item in sidebar */
    .chat-item {
        display: flex;
        align-items: center;
        justify-content: flex-start; /* Align items to the start */
        gap: 0.5rem; /* Add space between items */
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.75rem;
        cursor: pointer;
        transition: background-color 0.2s;
        color: var(--text-color); /* Use theme text color */
        border: 1px solid transparent; /* Default transparent border */
    }
    .chat-item:hover {
        background-color: #e0e0e0; /* Light hover */
    }
    /* Remove dark-mode specific hover */
    /* .dark-mode .chat-item:hover {
        background-color: #2a2a4a;
    } */
    .chat-item.active { background-color: #1f1f1f; border: 1px solid transparent; font-weight: 600; }
    .chat-item-text {
        flex-grow: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; /* Add ellipsis for long text */
    }
    .chat-item-actions {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        margin-left: auto; /* Push actions to the far right */
        flex-shrink: 0; /* Prevent shrinking */
    }
    .chat-item-delete {
        color: var(--delete-icon-color);
        cursor: pointer;
        font-size: 1.1em;
        opacity: 0.7;
        transition: opacity 0.2s;
        padding: 0.25rem;
    }
    .chat-item-delete:hover {
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- JavaScript for Clipboard Copy (placed here for global availability) ---
# This script will be executed once when the app loads.
st.markdown(
    """
    <script>
    function copyTextToClipboard(text) {
        if (navigator.clipboard) { // Use modern clipboard API if available
            navigator.clipboard.writeText(text).then(function() {
                // Send message back to Streamlit to show a toast
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Copied to clipboard!', toast_type: 'success' },
                        bubble: true
                    }
                }, '*');
            }, function(err) {
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Failed to copy!', toast_type: 'error' },
                        bubble: true
                    }
                }, '*');
            });
        } else { // Fallback for older browsers
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed"; // Avoid scrolling to bottom
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                var successful = document.execCommand('copy');
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Copied to clipboard (fallback)!', toast_type: 'info' },
                        bubble: true
                    }
                }, '*');
            } catch (err) {
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Failed to copy (fallback)!', toast_type: 'error' },
                        bubble: true
                    }
                }, '*');
            }
            document.body.removeChild(textArea);
        }
    }

    // Listener to receive messages from the JavaScript to trigger Streamlit toasts
    window.addEventListener('message', event => {
        if (event.data.streamlit && event.data.streamlit.command === 'SET_PAGE_STATE' && event.data.streamlit.state.toast_message) {
            const message = event.data.streamlit.state.toast_message;
            const type = event.data.streamlit.state.toast_type || 'info';
            // Streamlit's toast function is available in the global scope
            if (window.streamlit && window.streamlit.setToast) {
                window.streamlit.setToast(message, type);
            }
        }
    });
    </script>
    """, unsafe_allow_html=True
)

# --- Python-side Toast Listener (to receive messages from JS) ---
# This checks for a 'toast_message' in session_state set by the JS message
if 'toast_message' in st.session_state and st.session_state.toast_message:
    toast_type = st.session_state.get('toast_type', 'info')
    if toast_type == 'success':
        st.toast(st.session_state.toast_message, icon="✅")
    elif toast_type == 'error':
        st.toast(st.session_state.toast_message, icon="❌")
    elif toast_type == 'info':
        st.toast(st.session_state.toast_message, icon="ℹ️")
    else:
        st.toast(st.session_state.toast_message)
    # Clear the toast message after displaying
    del st.session_state.toast_message
    if 'toast_type' in st.session_state:
        del st.session_state.toast_type


# --- Dark Mode Toggle (using session state and CSS class) ---
# Removed dark mode toggle function as it's no longer needed
# def toggle_dark_mode():
#     st.session_state.dark_mode = not st.session_state.dark_mode
#     print(f"DEBUG: Dark mode toggled to {st.session_state.dark_mode}")
#     st.rerun() # Rerun to apply CSS class change


# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>DocuMind</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-actions'>", unsafe_allow_html=True)
    if st.button("➕ New chat", key="nav_new_chat", use_container_width=True):
        start_new_chat()
    if st.button("🔍 Search chats", key="toggle_search", use_container_width=True):
        st.session_state.show_search = not st.session_state.show_search
    if st.session_state.show_search:
        search_query = st.text_input("Search", key="chat_search", placeholder="Search chats")
    else:
        search_query = ""
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'><span>Features</span><span class='hr'></span></div>", unsafe_allow_html=True)

    # Dark Mode Toggle Button - REMOVED
    # if st.button(f"{'☀️ Light Mode' if st.session_state.dark_mode else '🌙 Dark Mode'}", key="dark_mode_toggle"):
    #     toggle_dark_mode()
    
    # st.markdown("---")

    

    # st.markdown("---")
    # st.header("Navigation")
    # Navigation buttons using st.button for reliability
    
    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    nav_container = st.container()
    with nav_container:
        # Summary Button
        summary_button_clicked = st.button("📄 Summary", key="nav_summary", use_container_width=True)
        if summary_button_clicked:
            if st.session_state.current_panel != 'summary':
                st.session_state.current_panel = 'summary'
                print("DEBUG: Switched to Summary panel.")
                st.rerun()

        # Ask Me Button
        ask_button_clicked = st.button("💬 Ask AI", key="nav_ask", use_container_width=True)
        if ask_button_clicked:
            if st.session_state.current_panel != 'ask':
                st.session_state.current_panel = 'ask'
                print("DEBUG: Switched to Ask Me panel.")
                st.rerun()

        # Challenge Me Button
        challenge_button_clicked = st.button("🧠 Challenge Me", key="nav_challenge", use_container_width=True)
        if challenge_button_clicked:
            if st.session_state.current_panel != 'challenge':
                st.session_state.current_panel = 'challenge'
                print("DEBUG: Switched to Challenge Me panel.")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='hr-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'><span>Past Chats</span><span class='hr'></span></div>", unsafe_allow_html=True)


    # Display list of past chats
    st.markdown("""
        <div class='sidebar-chats' style='flex-grow: 1; overflow-y: auto; padding-right: 10px;'>
    """, unsafe_allow_html=True)

    
    chats_source = st.session_state.chats if st.session_state.chats else []
    if search_query:
        sq = search_query.strip().lower()
        chats_source = [c for c in chats_source if sq in c.get("name", "").lower()]
    if chats_source:
        for chat in chats_source:
            chat_id = chat["chat_id"]
            chat_name = chat["name"]
            doc_name = chat["document_name"] if chat["document_name"] else "No Document"
            # Ensure timestamp exists before formatting
            chat_timestamp = datetime.fromisoformat(chat["timestamp"]).strftime('%Y-%m-%d %H:%M') if chat.get("timestamp") else "N/A"

            is_active = (chat_id == st.session_state.current_chat_id)
            
            # Use st.columns to layout the chat item and its action buttons
            # The main button takes up most of the space, and the action buttons are small
            col1, col2 = st.columns([0.88, 0.12])

            with col1:
                if st.button(f"{chat_name}", key=f"load_chat_{chat_id}", use_container_width=True):
                    chat_data_to_load = st.session_state.document_rag.load_chat_metadata(chat_id)
                    if chat_data_to_load:
                        load_chat_session(chat_data_to_load)
                    else:
                        st.error("Failed to load chat data.")

            with col2:
                if st.button("⋮", key=f"menu_chat_{chat_id}", use_container_width=True):
                    st.session_state[f"show_menu_{chat_id}"] = not st.session_state.get(f"show_menu_{chat_id}", False)

            if st.session_state.get(f"show_menu_{chat_id}", False):
                st.markdown("<div class='chat-menu'>", unsafe_allow_html=True)
                m1, m2 = st.columns([0.5, 0.5])
                with m1:
                    if st.button("Rename", key=f"menu_rename_{chat_id}", use_container_width=True):
                        st.session_state[f"renaming_{chat_id}"] = True
                        st.session_state[f"show_menu_{chat_id}"] = False
                with m2:
                    if st.button("Delete", key=f"menu_delete_{chat_id}", use_container_width=True):
                        delete_chat_session(chat_id)
                st.markdown("</div>", unsafe_allow_html=True)

            # Logic for renaming a chat
            if st.session_state.get(f"renaming_{chat_id}"):
                new_name = st.text_input("Rename", value=chat_name, key=f"rename_input_{chat_id}")
                rcol1, rcol2 = st.columns([1,1])
                if rcol1.button("Save", key=f"save_rename_{chat_id}"):
                    st.session_state.document_rag.update_chat_metadata(chat_id, {"name": new_name})
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_name = new_name
                    fetch_all_chats()
                    st.session_state[f"renaming_{chat_id}"] = False
                    st.rerun()
                if rcol2.button("Cancel", key=f"cancel_rename_{chat_id}"):
                    st.session_state[f"renaming_{chat_id}"] = False
    else:
        st.info("No past chats yet. Start a new one by uploading a document!")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr-line'></div>", unsafe_allow_html=True)
    st.header("Upload Document")
    
    # User ID is no longer directly from Firebase auth, can be removed or simplified
    # For local storage, a fixed identifier or simply omitting it is fine.
    # st.markdown(f"<p style='font-size:0.8em; opacity:0.6; margin-top: 20px;'>User ID: Local Storage</p>", unsafe_allow_html=True)


# Function to format chat history for download
def format_chat_history_for_download():
    """Formats the chat history into a plain text string."""
    history_string = []
    for i, chat_entry in enumerate(st.session_state.ask_history): # Iterate over dictionaries
        q = chat_entry.get('question', 'N/A')
        a = chat_entry.get('answer', 'N/A')
        j = chat_entry.get('justification', '')
        
        history_string.append(f"--- Conversation Turn {i+1} ---\n")
        history_string.append(f"You: {q}\n")
        history_string.append(f"Smart Assistant: {a}\n")
        if j:
            history_string.append(f"Justification: {j}\n")
        history_string.append("\n") # Add a blank line for readability between turns
    return "\n".join(history_string)


# --- Main Content Panel ---
main_content_container = st.container()
with main_content_container:
    st.markdown(f"""
        <h1 style="color: var(--text-color); font-size: 2.5rem; font-weight: 700; margin-bottom: 1.5rem;">
            {st.session_state.current_chat_name} - {st.session_state.current_panel.replace('_', ' ').title()}
        </h1>
    """, unsafe_allow_html=True)

    # Conditional display based on document presence and new chat flow
    if not st.session_state.full_document_text and (st.session_state.is_new_chat_flow or not st.session_state.uploaded_file_name):
        st.info("Start a new chat by uploading a document.")
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            center_uploader = st.file_uploader("", type=["pdf", "txt"], key="center_file_uploader", label_visibility="collapsed")
        if center_uploader:
            st.session_state.uploaded_file_name = center_uploader.name
            fd, temp_file_path = tempfile.mkstemp(suffix=f".{center_uploader.type.split('/')[-1]}")
            file_type = center_uploader.type.split('/')[-1]
            if file_type == 'plain':
                file_type = 'txt'
            try:
                with os.fdopen(fd, 'wb') as tmp_file:
                    tmp_file.write(center_uploader.getvalue())
                with st.spinner("Processing document and generating embeddings..."):
                    processed_chunks = process_document(temp_file_path, file_type)
                    if processed_chunks:
                        metadatas = [{"source": center_uploader.name, "chunk_index": i, "chat_id": st.session_state.current_chat_id} for i in range(len(processed_chunks))]
                        st.session_state.document_rag.add_documents(processed_chunks, chat_id=st.session_state.current_chat_id, metadatas=metadatas)
                        st.session_state.full_document_text = " ".join(processed_chunks)
                        with st.spinner("Generating summary..."):
                            summary = generate_summary_with_gemini(st.session_state.full_document_text)
                            st.session_state.summary = summary
                        save_chat_session(
                            st.session_state.current_chat_id,
                            st.session_state.current_chat_name,
                            st.session_state.uploaded_file_name,
                            st.session_state.full_document_text,
                            st.session_state.summary,
                            st.session_state.ask_history,
                            len(processed_chunks)
                        )
                        fetch_all_chats()
                        st.session_state.current_panel = 'summary'
                        st.session_state.is_new_chat_flow = False
                        st.rerun()
                    else:
                        st.error("Failed to process document.")
                        st.session_state.uploaded_file_name = None
                        st.session_state.full_document_text = ""
                        st.session_state.summary = ""
            except Exception as e:
                st.error(f"Error saving or processing uploaded file: {e}")
                st.session_state.uploaded_file_name = None
                st.session_state.full_document_text = ""
                st.session_state.summary = ""
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    else: # Document is loaded for the current chat
        if st.session_state.current_panel == 'summary':
            st.markdown(f"""
                <div style="background-color: var(--panel-bg); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border-color); flex-grow: 1; overflow-y: auto;">
                    <p style="color: var(--text-color); white-space: pre-wrap;">{st.session_state.summary}</p>
                </div>
            """, unsafe_allow_html=True)

        elif st.session_state.current_panel == 'ask':
            # Add download chat history button at the top of the chat panel
            if st.session_state.ask_history:
                st.download_button(
                    label="Download Chat History",
                    data=format_chat_history_for_download(),
                    file_name=f"{st.session_state.current_chat_name.replace(' ', '_')}_chat_history.txt",
                    mime="text/plain",
                    key="download_chat_history",
                    help="Download the entire conversation history as a text file."
                )
                st.markdown("---") # Separator

            # Use a container for chat history to enable scrolling
            chat_history_container = st.container(height=400) # Fixed height for scrollable chat
            with chat_history_container:
                for chat_entry in st.session_state.ask_history: # Iterate over dictionaries
                    q = chat_entry.get('question', 'N/A')
                    a = chat_entry.get('answer', 'N/A')
                    j = chat_entry.get('justification', '')

                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <p><strong>You:</strong> {q}</p>
                        </div>
                        <div class="chat-message ai-message">
                            <p><strong>DocuMind:</strong> {a}</p>
                            {f'<p style="font-size: 0.85em; opacity: 0.8; margin-top: 5px;"><strong>Justification:</strong> {j}</p>' if j else ''}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Input for new question
            user_question = st.text_input(
                "Ask a question about the document:",
                value=st.session_state.ask_question_input_value, # Controlled input
                key="ask_question_input",
                placeholder="e.g., What are the main findings of the research?"
            )
            
            if st.button("Ask", key="ask_button"):
                if user_question.strip():
                    with st.spinner("Fetching answer..."):
                        # Retrieve relevant context from ChromaDB for the current chat_id
                        relevant_chunks = st.session_state.document_rag.query_documents(
                            user_question, 
                            chat_id=st.session_state.current_chat_id, # Pass the current chat_id for filtering
                            n_results=5
                        )
                        
                        if not relevant_chunks:
                            st.warning("No highly relevant context found for your question. The answer might be general or indicate the information isn't directly in the document.")
                            
                        # Generate response using Gemini
                        answer, justification = generate_response_with_gemini(user_question, relevant_chunks)
                        
                        # Append to current chat's history in session state
                        st.session_state.ask_history.append({'question': user_question, 'answer': answer, 'justification': justification})
                        st.session_state.ask_question_input_value = "" # Clear the input box after submission
                        
                        # More efficient: only update the ask history, not the whole document
                        update_ask_history(
                            st.session_state.current_chat_id,
                            st.session_state.ask_history, # Save the updated history
                        )
                        st.rerun() # Rerun to update chat history display
                else:
                    st.warning("Please enter a question.")

        elif st.session_state.current_panel == 'challenge':
            challenge_container = st.container(height=600) # Fixed height for scrollable challenge section
            with challenge_container:
                if st.button("Generate New Challenge Questions", key="generate_questions_button"):
                    with st.spinner("Generating challenge questions..."):
                        st.session_state.challenge_questions = generate_challenge_questions(st.session_state.full_document_text, num_questions=3)
                        st.session_state.user_answers = [""] * len(st.session_state.challenge_questions) # Initialize empty answers
                        st.session_state.evaluation_results = [None] * len(st.session_state.challenge_questions) # Initialize empty evaluation results
                    st.success("Challenge questions generated!")
                    st.rerun() # Rerun to display questions

                if not st.session_state.challenge_questions:
                    st.info("Click 'Generate New Challenge Questions' to get started.")
                else:
                    st.markdown("### Answer the following questions based on the document:")
                    
                    for i, q in enumerate(st.session_state.challenge_questions):
                        st.markdown(f"""
                            <div style="background-color: var(--panel-bg); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border-color); margin-bottom: 1.5rem;">
                                <p style="font-weight: 600; font-size: 1.1em; color: var(--text-color); margin-bottom: 0.75rem;">Question {i+1}: {q}</p>
                        """, unsafe_allow_html=True)

                        # Visible answer box label and input below each question
                        answer_box = st.container()
                        with answer_box:
                            st.session_state.user_answers[i] = st.text_area(
                                "",
                                value=st.session_state.user_answers[i],
                                key=f"user_answer_{i}",
                                height=120,
                                placeholder="Type your answer here..."
                            )
                        
                        if st.button(f"Evaluate Answer for Q{i+1}", key=f"evaluate_btn_{i}"):
                            if st.session_state.user_answers[i].strip():
                                with st.spinner("Evaluating your answer..."):
                                    is_correct, justification, score, desired_answer_snippet = evaluate_user_answer(
                                        q, 
                                        st.session_state.user_answers[i], 
                                        st.session_state.full_document_text
                                    )
                                    evaluation_feedback = {
                                        "is_correct": is_correct,
                                        "justification": justification,
                                        "score": score,
                                        "desired_answer_snippet": desired_answer_snippet
                                    }
                                    st.session_state.evaluation_results[i] = evaluation_feedback
                                    st.success("Evaluation complete!")
                                    st.rerun() # Rerun to display evaluation results
                            else:
                                st.warning("Please provide an answer to evaluate.")
                        
                        # Display evaluation results if available
                        if st.session_state.evaluation_results[i]:
                            eval_res = st.session_state.evaluation_results[i]
                            # Ensure desired_answer_snippet is a string, even if None or empty
                            snippet_to_copy = eval_res['desired_answer_snippet'] if eval_res['desired_answer_snippet'] else ""
                            st.markdown(f"""
                                <div style="margin-top: 1rem; padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color); background-color: var(--input-bg);">
                                    <p style="font-weight: 700; color: {'#2e7d32' if eval_res['is_correct'] else '#c62828'};">
                                        Evaluation: {'Correct' if eval_res['is_correct'] else 'Incorrect'}
                                    </p>
                                    <p style="font-weight: 600; color: var(--text-color); margin-top: 0.5rem;">Score: {eval_res['score']}/10</p>
                                    <p style="color: var(--text-color); margin-top: 0.5rem;">
                                        <strong>Justification:</strong> {eval_res['justification']}
                                    </p>
                                    {f'''
                                    <p style="color: var(--text-color); margin-top: 0.5rem; display: inline-block;">
                                        <strong>Desired Answer Snippet:</strong> {snippet_to_copy}
                                    </p>
                                    <span class="copy-icon" onclick="copyTextToClipboard(`{snippet_to_copy.replace('`', '\\`')}`)">📋</span>
                                    ''' if snippet_to_copy and snippet_to_copy != "N/A" else ''}
                                </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True) # End of question container