from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self.model_name = model_name

    @property
    def model(self):
        """Load model lazily."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """Rerank chunks by relevance to the query."""
        if not chunks:
            return []

        # Score each chunk against the query
        pairs = [(query, doc.page_content) for doc in chunks]
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_chunks = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [chunk for chunk, score in scored_chunks[:top_k]]
