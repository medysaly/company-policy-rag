from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document


def test_load_text_file():
    docs = load_document("data/sample/rag_explained.txt")
    assert len(docs) == 1
    assert "RAG" in docs[0].page_content


def test_chunks_are_created():
    docs = load_document("data/sample/rag_explained.txt")
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunk_size_within_limit():
    docs = load_document("data/sample/rag_explained.txt")
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_chunk_metadata_preserved():
    docs = load_document("data/sample/rag_explained.txt")
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "chunk_id" in chunk.metadata


def test_unsupported_file_raises_error():
    try:
        load_document("data/sample/fake.csv")
        assert False, "Should have raised an error"
    except (ValueError, FileNotFoundError):
        pass
