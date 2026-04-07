from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.generation.llm import get_llm
from app.vectorstore.store import get_vector_store


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful company policy assistant. Answer the question based ONLY on the following context from company documents.

If the context does not contain enough information to answer the question, say "I don't have enough information in the company documents to answer that."

Always be specific and cite which document the information comes from when possible.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGChain:
    def __init__(self):
        self.llm = get_llm()
        self.store = get_vector_store()
        self.parser = StrOutputParser()

    def ingest(self, chunks: list):
        """Add document chunks to the vector store."""
        self.store.add_documents(chunks)

    def query(self, question: str, top_k: int = 5) -> dict:
        """Ask a question and get an answer from the documents."""
        # 1. Retrieve relevant chunks
        results = self.store.similarity_search(question, k=top_k)

        # 2. Format context from chunks
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in results
        )

        # 3. Build prompt and call LLM
        chain = RAG_PROMPT | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})

        # 4. Return answer with sources
        return {
            "answer": answer,
            "sources": [
                {
                    "text": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                }
                for doc in results
            ],
        }
