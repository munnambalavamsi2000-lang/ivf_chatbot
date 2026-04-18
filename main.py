import os
import shutil
from typing import List
from dotenv import load_dotenv  

# -----------------------------
# 0️⃣ TEMPORARY: Set HF_TOKEN directly
# -----------------------------
# Replace this with your actual HuggingFace token
load_dotenv()  # loads variables from .env into os.environ

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")
# -----------------------------
# 1️⃣ CONFIGURATION
# -----------------------------
PDF_PATH = r"C:\Ai2_Project\data_pdf"  
PERSIST_DIR = "./ivf_chroma_db"

# -----------------------------
# 2️⃣ CLEAN OLD VECTOR DATABASE
# -----------------------------
if os.path.exists(PERSIST_DIR):
    print("🧹 Removing old Chroma DB...")
    shutil.rmtree(PERSIST_DIR)

# -----------------------------
# 3️⃣ EMBEDDINGS
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

print("🔹 Loading miniLM embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# 4️⃣ LOAD & CHUNK DOCUMENTS
# -----------------------------
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("📄 Loading IVF PDFs...")
loader = PyPDFDirectoryLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

# -----------------------------
# 5️⃣ VECTOR STORE (CHROMA)
# -----------------------------
from langchain_community.vectorstores import Chroma

print("📦 Creating Chroma Vector Store...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
vector_db.persist()

# -----------------------------
# 6️⃣ LLM SETUP (Mistral-7B)
# -----------------------------
print("🤖 Loading Mistral-7B Instruct...")
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=600
)

chat_model = ChatHuggingFace(llm=llm)

# -----------------------------
# 7️⃣ ADVANCED PROMPT (Medical-safe + Empathetic)
# -----------------------------
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
<s>[INST]
You are an IVF Patient Support Assistant.

Rules:
- Answer ONLY from the provided context
- Be empathetic and easy to understand
- If medical advice is required, suggest consulting a fertility specialist
- Do NOT hallucinate

Context:
{context}

User Question:
{question}
[/INST]
</s>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

# -----------------------------
# 8️⃣ RETRIEVAL QA (MMR search)
# -----------------------------
#from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA



qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vector_db.as_retriever(
        search_type="mmr",   # More advanced than simple similarity
        search_kwargs={"k": 5, "fetch_k": 20}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# 9️⃣ TEST QUERY
# -----------------------------
query = "What are the common side effects of IVF stimulation medications?"

response = qa_chain.invoke({"query": query})

print("\n💬 Chatbot Answer:\n")
print(response["result"])

print("\n📚 Sources Used:\n")
for doc in response["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown"))