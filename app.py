# app.py

from __future__ import annotations
import streamlit as st

# Import custom modules
from schemas import ResumeAnalysisSchema
from document_loader import load_pdf_with_langchain
from llm_engine import run_analysis_chain, run_rag_recommendations
from rag_store import initialize_vector_store

# Page Setup
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🦜", layout="wide")

st.title("🦜 AI Resume Analyzer & RAG Skill Agent")
st.caption("Powered by LangChain, Groq LLaMA 3.3, Pydantic, & ChromaDB Vector Store")

# Configuration Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get a free API key at console.groq.com")

# Main Input Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Upload Candidate CV")
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area("Paste Target Job Requirements...", height=200)

# Action Trigger
if st.button("🚀 Analyze with LangChain Agent", type="primary"):
    if not groq_api_key:
        st.error("Please provide a Groq API Key in the sidebar!")
    elif not uploaded_pdf or not job_description.strip():
        st.warning("Please upload a PDF and paste a job description.")
    else:
        with st.spinner("Executing LangChain Ingestion & RAG Pipeline..."):
            try:
                # Initialize Vector DB
                vectorstore = initialize_vector_store()
                
                # Step 1: Parse PDF
                raw_cv_text = load_pdf_with_langchain(uploaded_pdf)
                
                # Step 2: Run LLM Evaluation Chain
                analysis_res: ResumeAnalysisSchema = run_analysis_chain(raw_cv_text, job_description, groq_api_key)
                
                st.divider()
                st.success("Analysis Complete!")
                
                # Render Metrics & Skills
                st.metric(label="🎯 ATS Score Match", value=f"{analysis_res.match_score}%")
                
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.subheader("✅ Skills Matched")
                    for s in analysis_res.found_skills:
                        st.write(f"• {s}")
                        
                with r_col2:
                    st.subheader("❌ Missing Skills Detected")
                    for s in analysis_res.missing_skills:
                        st.write(f"• {s}")
                        
                st.divider()
                
                # Step 3: Run RAG Search for Missing Skills
                st.subheader("📚 RAG Learning Roadmaps (Retrieved from ChromaDB)")
                if analysis_res.missing_skills:
                    rag_results = run_rag_recommendations(analysis_res.missing_skills, vectorstore, groq_api_key)
                    for item in rag_results:
                        with st.expander(f"📖 Study Roadmap for: **{item['skill']}**"):
                            st.write(item['advice'])
                            st.caption("Database Sources:")
                            for doc in item['sources']:
                                st.write(f"- *{doc.page_content}* | **Resource:** `{doc.metadata.get('resource')}`")
                else:
                    st.info("No critical missing skills found! Great match.")
                    
            except Exception as e:
                st.error(f"Execution Error: {e}")