import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Company Policy RAG", page_icon="📋", layout="wide")
st.title("📋 Company Policy Q&A")
st.caption("Upload company documents and ask questions about policies")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt", "md"],
    )

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = httpx.post(f"{API_URL}/ingest", files=files, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Ingested {result['num_chunks']} chunks!")
                else:
                    st.error(f"Error: {response.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Is the server running? (make serve)")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": top_k},
                    timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result["answer"])

                    # Show sources
                    with st.expander("📄 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"**{source['source']}**")
                            st.text(source["text"])
                            st.divider()

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                else:
                    st.error(f"Error: {response.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Is the server running? (make serve)")
