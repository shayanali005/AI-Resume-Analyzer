# document_loader.py

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_with_langchain(uploaded_file) -> str:
    """Uses LangChain PyPDFLoader to read unstructured PDF resumes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)  # Cleanup temp file
    
    return "\n".join([doc.page_content for doc in docs])