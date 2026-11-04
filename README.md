# 🎥 YouTube AI Agent (RAG + Groq)

An intelligent AI Agent built with **FastAPI**, **ChromaDB**, and **Groq API** that allows users to search or provide any YouTube video, chat with its content in real-time, generate full summaries, and jump to precise timestamps.

---

## 🎯 Doctor / Project Requirements Met

- **LLM Engine**: Groq API using model `openai/gpt-oss-120b` (with automatic fallback to `llama-3.3-70b-versatile`).
- **AI Agent Capabilities**:
  - Accept YouTube video links and direct search queries.
  - Interactive chat with the video transcript.
  - Strict guardrails: polite acknowledgment for greetings/thanks, and courteous refusal for out-of-scope questions.
- **Backend**: FastAPI with async support, CORS, and modular endpoints.
- **Frontend**: Clean and responsive HTML5, CSS3, and modern JavaScript with embedded video player and markdown rendering.
- **RAG Architecture**: ChromaDB vector store with multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and timestamp-aware retrieval.

---

## 🚀 Key Features

1. **Direct Search & Processing**: Paste any YouTube link or search by keywords directly from the interface.
2. **Timestamp-Aware Retrieval (Temporal RAG)**: Ask about specific minutes (e.g., *"What did the speaker mention at minute 10?"*) and receive accurate context.
3. **Comprehensive Video Analysis**: Automatic full-video overview, key points, and chronological table of contents.
4. **Interactive Jump-to-Time**: Clickable timestamp tags in chat that immediately jump the embedded YouTube player to that moment.
5. **Local History & Caching**: Fast local retrieval for previously processed videos without re-fetching transcripts.

---

## 📁 Project Structure

```text
first_assiment_first_project/
│
├── main.py                     # FastAPI backend & RAG pipeline
├── youtube_rag_assistant.ipynb  # Interactive development & experimentation notebook
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git privacy & security rules
├── README.md                   # Documentation
│
└── static/
    └── index.html              # Frontend UI (HTML5, CSS3, JS + Marked.js)