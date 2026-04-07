from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


def get_embeddings():
    """Create and return the embedding model."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
