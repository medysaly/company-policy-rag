# Architecture Decision Records

## ADR-001: Qdrant over ChromaDB/Pinecone

**Status:** Accepted

**Context:** We needed a vector database for storing and searching document embeddings.

**Decision:** Qdrant with in-memory mode for development and Docker for production.

**Reasoning:**
- Native hybrid search support (dense + sparse vectors) — critical for Step 7
- Free local mode with no server setup required
- Docker image for production deployment
- ChromaDB lacks native hybrid search
- Pinecone requires an account and has free tier limits

## ADR-002: Local Embeddings over OpenAI API

**Status:** Accepted

**Context:** We needed an embedding model to convert text into vectors.

**Decision:** sentence-transformers (all-MiniLM-L6-v2) running locally.

**Reasoning:**
- Free — no API costs during development or evaluation
- Fast — runs locally without network latency
- No API key required — reduces setup friction
- 384 dimensions is sufficient for document-level retrieval
- Trade-off: slightly lower quality than OpenAI text-embedding-3-large, but acceptable for this use case

## ADR-003: Chunking Strategy

**Status:** Accepted

**Context:** Documents need to be split into smaller pieces for effective retrieval.

**Decision:** RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=200.

**Reasoning:**
- Recursive splitting respects natural text boundaries (paragraphs > sentences > words)
- 1000 characters captures roughly one complete thought or policy section
- 200 character overlap prevents information loss at chunk boundaries
- These values are configurable via environment variables for experimentation

## ADR-004: Hybrid Search with Reciprocal Rank Fusion

**Status:** Accepted

**Context:** Pure dense vector search can miss keyword-specific queries (e.g., acronyms like "ESPP").

**Decision:** Combine dense search (Qdrant) with sparse search (BM25) using RRF.

**Reasoning:**
- Dense search captures semantic meaning but may miss exact terms
- BM25 excels at keyword matching but misses semantic similarity
- RRF combines rankings without needing to normalize different score scales
- alpha parameter allows tuning the balance between dense and sparse

## ADR-005: Cross-Encoder Reranking

**Status:** Accepted

**Context:** Initial retrieval returns candidates that could be more precisely ranked.

**Decision:** Two-stage retrieval — retrieve more candidates, then rerank with a cross-encoder.

**Reasoning:**
- Cross-encoders (ms-marco-MiniLM-L-6-v2) score query-document pairs jointly, which is more accurate than separate embeddings
- Too slow to use on the full corpus, but fast enough for reranking 10-15 candidates
- Improves precision without increasing the embedding model size

## ADR-006: LangChain for Orchestration

**Status:** Accepted

**Context:** We needed tooling for document loading, text splitting, and LLM integration.

**Decision:** Use LangChain for document loading, chunking, embeddings, vector store, and LLM calls.

**Reasoning:**
- Most recognized AI framework — strong resume signal
- Unified interface across LLM providers (swap Claude for GPT with one config change)
- Mature document loaders for PDF, text, and other formats
- Active community and documentation
