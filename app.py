import streamlit as st

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.generation.rag_chain import RAGChain

import tempfile
import os


@st.cache_resource
def get_rag_chain():
    """Create RAG chain once and reuse across sessions."""
    chain = RAGChain()

    # Pre-load sample document
    sample_path = "data/sample/remote_work_policy.pdf"
    if os.path.exists(sample_path):
        docs = load_document(sample_path)
        chunks = chunk_documents(docs)
        chain.ingest(chunks)

    return chain


st.set_page_config(page_title="Company Policy RAG", page_icon="📋", layout="wide")
st.title("📋 Company Policy Q&A")
st.caption("Upload company documents and ask questions about policies")

rag = get_rag_chain()

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt", "md"],
    )

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                docs = load_document(tmp_path)
                chunks = chunk_documents(docs)
                rag.ingest(chunks)
                st.success(f"Ingested {len(chunks)} chunks!")
            finally:
                os.unlink(tmp_path)

    st.divider()
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']}**")
                    st.text(source["text"])
                    st.divider()

if prompt := st.chat_input("Ask a question about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.query(prompt, top_k=top_k)
            st.markdown(result["answer"])

            with st.expander("📄 Sources"):
                for source in result["sources"]:
                    st.markdown(f"**{source['source']}**")
                    st.text(source["text"])
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
