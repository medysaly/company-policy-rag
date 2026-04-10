import os
import tempfile

import streamlit as st

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.generation.rag_chain import RAGChain


@st.cache_resource
def get_rag_chain():
    """Create RAG chain once and reuse across sessions."""
    chain = RAGChain()

    sample_path = "data/sample/remote_work_policy.pdf"
    if os.path.exists(sample_path):
        docs = load_document(sample_path)
        chunks = chunk_documents(docs)
        chain.ingest(chunks)

    return chain


st.set_page_config(page_title="Company Policy RAG", page_icon="📋", layout="wide")

# Header
st.title("📋 Company Policy Q&A")
st.caption("Ask questions about company policy documents and get accurate, sourced answers")

rag = get_rag_chain()

# Sidebar
with st.sidebar:
    st.header("📄 Documents")

    st.success("✅ **Remote Work Policy** is pre-loaded. You can start asking questions right away!")

    st.divider()

    st.subheader("Upload Additional Documents")
    uploaded_file = st.file_uploader(
        "Add your own PDF or TXT file",
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
                st.success(f"Ingested {len(chunks)} chunks from {uploaded_file.name}!")
            finally:
                os.unlink(tmp_path)

    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

    st.divider()

    st.markdown("**Built with:**")
    st.markdown("LangChain · Qdrant · Claude · FastAPI")
    st.markdown("[GitHub Repo](https://github.com/medysaly/company-policy-rag)")

# Sample questions
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("### Try asking:")
    cols = st.columns(2)
    sample_questions = [
        "Who needs to approve a remote work request?",
        "Is remote work a substitute for childcare?",
        "What happens to workers' compensation when working from home?",
        "Can the Town terminate a remote work agreement?",
    ]
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(q, key=f"sample_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']}**")
                    st.text(source["text"])
                    st.divider()

# Chat input
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
