import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load sample documents on startup."""
    from app.generation.rag_chain import rag_chain
    from app.ingestion.chunker import chunk_documents
    from app.ingestion.loader import load_document

    sample_path = "data/sample/remote_work_policy.pdf"
    if os.path.exists(sample_path):
        print("Loading sample document...")
        docs = load_document(sample_path)
        chunks = chunk_documents(docs)
        rag_chain.ingest(chunks)
        print(f"Pre-loaded {len(chunks)} chunks from {sample_path}")

    yield


app = FastAPI(
    title="Company Policy RAG",
    description="Ask questions about company policy documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
