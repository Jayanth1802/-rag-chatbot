import os
import shutil
import streamlit as st
from ingest import ingest_uploaded_files
from chain import load_qa_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")
st.title("RAG Document Q&A")
st.caption("Upload your documents and ask anything from them")

uploaded_files = st.file_uploader(
    "Upload your files (PDF, PPTX, DOCX)",
    type=["pdf", "pptx", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.cache_resource.clear()

        success = False
        with st.spinner("Reading and indexing your documents..."):
            try:
                success = ingest_uploaded_files(uploaded_files)
            except Exception as e:
                st.error(f"Could not process files: {e}")

        if success:
            st.success(f"Done! {len(uploaded_files)} file(s) processed. Ask your questions below.")
            st.session_state.messages = []

if os.path.exists("faiss_index"):
    @st.cache_resource
    def get_chain():
        return load_qa_chain()

    try:
        chain, retriever = get_chain()
    except Exception as e:
        st.error(f"Could not load the Q&A chain: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question from your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = chain.invoke(prompt)
                    sources = retriever.invoke(prompt)
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")
                    st.stop()

            # Handle both string and dict responses safely
            if isinstance(result, dict):
                answer = result.get("result") or result.get("answer") or str(result)
            else:
                answer = result

            st.markdown(answer)

            with st.expander("Sources used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** — {doc.metadata.get('source', 'unknown')}")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload documents and click 'Process Documents' to get started.")