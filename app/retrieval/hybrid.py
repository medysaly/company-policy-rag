from rank_bm25 import BM25Okapi

from app.vectorstore.store import get_vector_store


class HybridRetriever:
    def __init__(self, store=None):
        self.store = store or get_vector_store()
        self.chunks = []
        self.bm25 = None

    def add_documents(self, chunks: list):
        """Store chunks for both dense and sparse search."""
        self.chunks = chunks
        self.store.add_documents(chunks)

        # Build BM25 index for sparse search
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, top_k: int) -> list:
        """Vector similarity search."""
        return self.store.similarity_search(query, k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> list:
        """BM25 keyword search."""
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [self.chunks[i] for i in top_indices]

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list:
        """Hybrid search combining dense and sparse results using RRF."""
        dense_results = self._dense_search(query, top_k=top_k * 2)
        sparse_results = self._sparse_search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        rrf_scores = {}
        k = 60  # RRF constant

        for rank, doc in enumerate(dense_results):
            doc_id = doc.page_content[:100]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + rank + 1)

        for rank, doc in enumerate(sparse_results):
            doc_id = doc.page_content[:100]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)

        # Sort by RRF score and return top_k
        all_docs = {doc.page_content[:100]: doc for doc in dense_results + sparse_results}
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        return [all_docs[doc_id] for doc_id in sorted_ids]
