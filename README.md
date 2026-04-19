# 👶 IVF Patient Support Chatbot (RAG + Streamlit)

## 📌 Overview

This project is an **AI-powered IVF Patient Support Chatbot** that answers user queries using **Retrieval-Augmented Generation (RAG)**.

It uses:

* 📄 IVF medical PDFs as knowledge base
* 🔍 Vector search (ChromaDB)
* 🤖 LLM (Mistral-7B via HuggingFace)
* 🌐 Streamlit web app with chat + voice input

---

## 🚀 Features

* 📚 PDF-based knowledge retrieval
* 🧠 Context-aware answers using RAG
* 💬 Chat interface with history support
* 🎙️ Voice input (Speech-to-Text)
* 🎨 Custom UI with background styling
* ⚡ Fast embeddings using MiniLM
* 🧾 Source-based answering (no hallucination prompt)

---

## 🏗️ Architecture

```id="z9ldbb"
User Query → Retriever (ChromaDB) → Relevant Chunks → LLM (Mistral-7B) → Answer
```

---

## 📂 Project Structure

```id="jgxvqh"
ivf_chatbot/
│
├── main.py                  # Build vector DB from PDFs
├── app.py                   # Streamlit chatbot app
├── ivf_chroma_db/           # Vector database (auto-generated)
├── data_pdf/                # IVF PDF documents
├── .env                     # HuggingFace API token
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repo

```bash id="w08y4l"
git clone https://github.com/your-username/ivf-chatbot.git
cd ivf-chatbot
```

---

### 2️⃣ Create virtual environment

```bash id="k1p5a3"
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash id="1mpfg3"
pip install -r requirements.txt
```

---

### 4️⃣ Setup environment variables

Create a `.env` file:

```env id="33vnmq"
HF_TOKEN=your_huggingface_token_here
```

---

## 📄 Step 1: Build Vector Database

Run:

```bash id="22e0nl"
python main.py
```

✔ Loads PDFs
✔ Splits text into chunks
✔ Generates embeddings
✔ Stores in ChromaDB

---

## 🌐 Step 2: Run Streamlit App

```bash id="n9xf1h"
streamlit run app.py
```

Open:

```id="5c7h8w"
http://localhost:8501
```

---

## 🎤 Voice Feature

* Click 🎙️ button
* Speak your question
* Automatically converted to text

---

## 💡 Example Questions

* What are IVF stimulation side effects?
* How does AMH affect fertility?
* What is the IVF process timeline?
* What precautions should be taken during IVF?

---

## 🧠 Model Details

* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
* **LLM:** mistralai/Mistral-7B-Instruct-v0.2
* **Vector DB:** Chroma

---

## ⚠️ Important Notes

* Requires internet (for HuggingFace API)
* Make sure PDF path is correct in `main.py`
* Background image path must exist in `app.py`

---

## 🚧 Known Issues

* Python 3.14 not supported (use 3.10 / 3.11)
* Speech recognition depends on microphone permissions

---

## 🔐 Disclaimer

This chatbot is for **educational purposes only**.
It does **not replace professional medical advice**.
Always consult a qualified fertility specialist.

---

## 👨‍💻 Author

**Munnam Bala Vamsi**

---

## ⭐ Contribute

Feel free to fork, improve, and submit PRs!

---

## 📜 License

MIT License
