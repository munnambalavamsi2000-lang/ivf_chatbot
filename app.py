import os
import streamlit as st
import speech_recognition as sr
import base64
import speech_recognition as sr
import base64
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate



# --- 1. CONFIG & STYLING ---
load_dotenv()  # loads variables from .env into os.environ

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is not set. Please check your .env file.")
    st.stop()

PERSIST_DIR = "./ivf_chroma_db"
IMAGE_PATH = r"C:\Users\munna\Downloads\background.jpg"

st.set_page_config(page_title="IVF Patient Support", page_icon="👶", layout="wide")

# Helper function to inject local image as background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Adding an overlay to ensure text readability */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.4); /* Adjust opacity for darker/lighter background */
        z-index: -1;
    }}

    /* Fix Sidebar Button Squashing */
    [data-testid="stSidebar"] button {{
        width: 100% !important;
        white-space: nowrap !important;
        margin-bottom: 10px !important;
        justify-content: flex-start !important;
        padding: 10px !important;
        background-blur: 10px !important;
    }}

    /* Glassmorphism for Messages */
    [data-testid="stChatMessage"] {{
        background-color: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px);
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    /* Input Area Styling */
    .stChatInputContainer {{
        padding-bottom: 20px !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the background function
if os.path.exists(IMAGE_PATH):
    set_background(IMAGE_PATH)
else:
    st.sidebar.error("Background image not found at path.")

# --- 2. SPEECH TO TEXT FUNCTION ---
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Listening...", icon="🎙️")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=8)
            return r.recognize_google(audio)
        except Exception:
            st.error("Could not hear you. Check mic settings.")
            return None

# --- 3. MODEL LOADING (Cached) ---
@st.cache_resource
def init_qa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HF_TOKEN, temperature=0.2)
    chat_model = ChatHuggingFace(llm=llm)
    
    prompt = PromptTemplate(
        template="<s>[INST] You are an IVF Support Assistant. Context: {context} \nQuestion: {question} [/INST] </s>",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=chat_model, chain_type="stuff", 
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = init_qa()

# --- 4. SESSION STATE ---
if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = {"Main Chat": []}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Main Chat"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title(" IVF Assistant")
    
    if st.button("➕ Start New Chat"):
        new_id = f"Chat {len(st.session_state.chat_history_list) + 1}"
        st.session_state.chat_history_list[new_id] = []
        st.session_state.active_chat = new_id
        st.rerun()

    st.markdown("### Chat History")
    st.session_state.active_chat = st.selectbox(
        "Select Session", 
        options=list(st.session_state.chat_history_list.keys()),
        index=list(st.session_state.chat_history_list.keys()).index(st.session_state.active_chat)
    )

    if st.button("🗑️ Clear This History"):
        st.session_state.chat_history_list[st.session_state.active_chat] = []
        st.rerun()

# --- 6. MAIN CHAT AREA ---
st.title("👶 IVF Patient Support")
st.write("Welcome. Ask me anything about medications, procedures, or IVF protocols.")

# Message Display
chat_placeholder = st.container()

# Voice/Text input row
input_row = st.container()
with input_row:
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        voice_btn = st.button("🎙️", help="Click to speak")
    with col2:
        text_input = st.chat_input("Message the IVF bot...")

# Handle Logic
user_query = None
if voice_btn:
    user_query = get_voice_input()
elif text_input:
    user_query = text_input

if user_query:
    st.session_state.chat_history_list[st.session_state.active_chat].append({"role": "user", "content": user_query})
    
    with st.spinner("Searching medical documents..."):
        response = qa_chain.invoke({"query": user_query})
        answer = response["result"]
        st.session_state.chat_history_list[st.session_state.active_chat].append({"role": "assistant", "content": answer})

# Render the history
with chat_placeholder:
    for msg in st.session_state.chat_history_list[st.session_state.active_chat]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])