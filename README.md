# 🦜 AI Resume Analyzer & RAG Skill Agent

An intelligent Applicant Tracking System (ATS) and skill gap analysis tool powered by **LangChain**, **Groq LLaMA 3.3**, **Pydantic**, **ChromaDB**, and **Streamlit**.

The system evaluates candidate resumes against job descriptions to provide instant ATS match scoring, identify matched and missing competencies, and retrieve personalized learning roadmaps using Retrieval-Augmented Generation (RAG).

---

## ✨ Key Features
- **PDF Resume Parsing:** Ingests unstructured resume PDFs using LangChain's `PyPDFLoader`.
- **ATS Match Scoring & Skill Extraction:** Evaluates candidate fit against job requirements using Groq LLaMA 3.3 via LangChain Expression Language (LCEL).
- **Structured Data Validation:** Uses Pydantic schema validation (`method="json_mode"`) to ensure strict JSON output.
- **RAG-Powered Learning Paths:** Queries an in-memory **ChromaDB** vector database with HuggingFace embeddings (`all-MiniLM-L6-v2`) to recommend curated study paths for missing skills.
- **Modular Architecture:** Cleanly separated codebase into distinct modules (`schemas.py`, `llm_engine.py`, `rag_store.py`, `document_loader.py`, `config.py`, and `app.py`).

---

## 🛠️ Tech Stack
- **Framework:** LangChain (LCEL)
- **LLM Provider:** Groq (`llama-3.3-70b-versatile`)
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace Transformers (`all-MiniLM-L6-v2`)
- **Schema Validation:** Pydantic v2
- **Document Ingestion:** PyPDF / LangChain Community
- **Frontend Dashboard:** Streamlit

---

## 📁 Repository Structure
```text
Resume_Analyzer/
├── config.py           # Global system constants & model settings
├── schemas.py          # Pydantic models with alias validation
├── document_loader.py  # LangChain PDF parsing routines
├── llm_engine.py       # LCEL analysis and RAG recommendation chains
├── rag_store.py        # ChromaDB vector store initialization
├── app.py              # Main Streamlit dashboard application
└── requirements.txt    # Project dependencies
