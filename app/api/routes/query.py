from fastapi import APIRouter

from app.api.schemas import QueryRequest, QueryResponse, Source

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    from app.generation.rag_chain import rag_chain

    result = rag_chain.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=result["answer"],
        sources=[
            Source(text=s["text"], source=s["source"])
            for s in result["sources"]
        ],
    )
