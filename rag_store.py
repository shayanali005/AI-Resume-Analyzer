# rag_store.py

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import EMBEDDING_MODEL_NAME, VECTOR_COLLECTION_NAME

@st.cache_resource
def initialize_vector_store():
    """Builds an in-memory ChromaDB vector store populated with course resources."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    knowledge_docs = [
        Document(page_content="LangGraph: Master stateful multi-agent workflows and graph-based AI orchestration.", metadata={"topic": "LangGraph", "resource": "LangChain Academy / DeepLearning.AI"}),
        Document(page_content="Docker & Containers: Learn containerization, multi-stage builds, Docker Compose, and Kubernetes basics.", metadata={"topic": "Docker", "resource": "Docker Docs & KodeKloud"}),
        Document(page_content="FastAPI: Production-ready Async Python APIs, OpenAPI generation, and Pydantic validation.", metadata={"topic": "FastAPI", "resource": "FastAPI Official Tutorial"}),
        Document(page_content="Vector Databases & Embeddings: Learn ChromaDB, Pinecone, FAISS, and semantic similarity search.", metadata={"topic": "ChromaDB", "resource": "Pinecone Learning Center"}),
        Document(page_content="PyTorch & Deep Learning: Neural networks, Transformers architecture, gradient descent, and fine-tuning models.", metadata={"topic": "PyTorch", "resource": "PyTorch 60-min Blitz"}),
        Document(page_content="MLOps & CI/CD: MLflow, GitHub Actions, model tracking, and cloud deployment on AWS/GCP.", metadata={"topic": "MLOps", "resource": "Made With ML Guide"}),
    ]
    
    vectorstore = Chroma.from_documents(
        documents=knowledge_docs,
        embedding=embeddings,
        collection_name=VECTOR_COLLECTION_NAME
    )
    return vectorstore