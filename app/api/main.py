from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingest, query

app = FastAPI(
    title="Company Policy RAG",
    description="Ask questions about company policy documents",
    version="0.1.0",
)

# Allow Streamlit (or any frontend) to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
