from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.services.retrieval import get_hybrid_context, format_prompt_context

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/api/query/retrieve")
def retrieve_hybrid_context(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Takes a natural language query, vectorizes it, fetches top document chunks 
    from PostgreSQL using pgvector, and traverses the Apache AGE Graph to fetch
    directly related medical entities (Disease, Drugs, Symptoms).
    """
    try:
        # Fetch the results from both DB topologies
        hybrid_results = get_hybrid_context(db, req.query, req.top_k)
        
        # Format the assembled context for an LLM
        formatted_prompt = format_prompt_context(hybrid_results)
        
        return {
            "status": "success",
            "query": req.query,
            "raw_results": hybrid_results,
            "formatted_prompt": formatted_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from backend.services.llm import generate_response

@router.post("/api/query/generate")
async def generate_hybrid_answer(req: QueryRequest, db: Session = Depends(get_db)):
    """
    End-to-end RAG workflow endpoint:
    1. Retrieves hybrid context (Vector + Graph)
    2. Sends context + query to local SLM (Ollama)
    3. Returns the synthesized answer
    """
    try:
        # 1. Retrieve hybrid context
        hybrid_results = get_hybrid_context(db, req.query, req.top_k)
        formatted_prompt = format_prompt_context(hybrid_results)
        
        # 2. Generate answer
        answer = await generate_response(req.query, formatted_prompt)
        
        return {
            "status": "success",
            "query": req.query,
            "answer": answer,
            "context_used": hybrid_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
