from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.models import Chunk
from backend.services.graph_extractor import build_chunk_graph
from backend.services.graph_storage import save_graph_to_db

router = APIRouter()

@router.post("/api/graph/generate/{document_id}")
def generate_knowledge_graph(document_id: int, db: Session = Depends(get_db)):
    """
    Retrieves all text chunks for a given document and generates 
    the Knowledge Graph entities & relationships for them.
    """
    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).all()
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this document")

    total_nodes = 0
    total_edges = 0
    
    for chunk in chunks:
        try:
            # 1. Extract entities and build NetworkX graph for this chunk
            G = build_chunk_graph(chunk.id, chunk.text_content)
            
            # 2. Persist the graph to PostgreSQL via Apache AGE
            stats = save_graph_to_db(db, G)
            total_nodes += stats.get("nodes", 0)
            total_edges += stats.get("edges", 0)
        except Exception as e:
            # Log error but continue with other chunks
            print(f"Error processing chunk {chunk.id}: {e}")
            continue

    return {
        "status": "success",
        "message": f"Graph generated for Document {document_id}",
        "nodes_extracted": total_nodes,
        "edges_extracted": total_edges
    }
