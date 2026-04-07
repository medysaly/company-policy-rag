from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    text: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    num_chunks: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
