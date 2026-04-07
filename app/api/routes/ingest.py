from fastapi import APIRouter, UploadFile, File
import tempfile
import os

from app.api.schemas import IngestResponse
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        # Load and chunk
        docs = load_document(tmp_path)
        chunks = chunk_documents(docs)

        # Store in vector store (import here to avoid circular imports)
        from app.generation.rag_chain import rag_chain
        rag_chain.ingest(chunks)

        return IngestResponse(
            num_chunks=len(chunks),
            status="success",
        )
    finally:
        os.unlink(tmp_path)  # Clean up temp file
