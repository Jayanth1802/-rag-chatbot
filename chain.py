import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_PATH = "faiss_index"


def get_secret(key):
    """Safely read a secret from Streamlit secrets, falling back to None
    if no secrets.toml exists at all (common in local dev)."""
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    api_key = get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to a .env file (GROQ_API_KEY=your_key) "
            "or to .streamlit/secrets.toml (GROQ_API_KEY = \"your_key\")."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        max_retries=5,
        timeout=60
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. You will be given context extracted from the user's uploaded documents.

Your job:
1. First check if the answer exists in the context below.
2. If YES — answer using the context and start with: "Based on your documents:"
3. If NO — answer from your own knowledge and start with: "⚠️ Not found in source. Based on general knowledge:"

Always be clear which one you are doing.

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever