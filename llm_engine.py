# llm_engine.py

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from schemas import ResumeAnalysisSchema
from config import GROQ_MODEL_NAME

def run_analysis_chain(cv_text: str, job_desc: str, api_key: str) -> ResumeAnalysisSchema:
    """LangChain LCEL Pipeline for structured ATS resume evaluation."""
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,
        groq_api_key=api_key
    )
    
    structured_llm = llm.with_structured_output(ResumeAnalysisSchema, method="json_mode")
    
    # Notice the double curly braces {{ and }} escaping the JSON example!
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert ATS (Applicant Tracking System) software and Technical Recruiter.\n"
         "Compare candidate CVs against Job Descriptions.\n"
         "You MUST output valid JSON containing EXACTLY these keys:\n"
         "{{\n"
         '  "match_score": 75,\n'
         '  "found_skills": ["Python", "Pandas"],\n'
         '  "missing_skills": ["Docker", "LangGraph"],\n'
         '  "recommendations": ["Add project examples for Docker", "Mention REST API experience"]\n'
         "}}"
        ),
        ("human", "Candidate CV:\n{cv_text}\n\nJob Description:\n{job_desc}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"cv_text": cv_text, "job_desc": job_desc})


def run_rag_recommendations(missing_skills: List[str], vectorstore, api_key: str):
    """Retrieves relevant courses from ChromaDB using LangChain vector search."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.2, groq_api_key=api_key)
    
    recommendations = []
    for skill in missing_skills[:4]:  # Query top 4 missing skills
        docs = retriever.invoke(f"How to learn {skill}")
        
        context_str = "\n".join([f"- {d.page_content} (Source: {d.metadata.get('resource')})" for d in docs])
        
        rag_prompt = ChatPromptTemplate.from_template(
            "Target Skill to Learn: {skill}\n"
            "Retrieved Context from Database:\n{context}\n\n"
            "Write a short, encouraging 2-sentence study plan for this candidate based on the retrieved context."
        )
        
        rag_chain = rag_prompt | llm | StrOutputParser()
        advice = rag_chain.invoke({"skill": skill, "context": context_str})
        recommendations.append({"skill": skill, "advice": advice, "sources": docs})
        
    return recommendations