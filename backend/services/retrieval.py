import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.services.embedding import generate_embedding

def get_hybrid_context(db: Session, query_text: str, top_k: int = 3) -> dict:
    """
    1. Embeds the user query.
    2. Finds the top_k most similar chunks using pgvector.
    3. For those chunks, traverses the Apache AGE graph to find connected entities.
    4. Returns a hybrid context object.
    """
    # 1. Vectorize the Query
    query_embedding = generate_embedding(query_text)
    
    # 2. Vector Search using pgvector
    vector_query = text("""
        SELECT id, document_id, text_content, embedding <=> :embedding AS distance
        FROM chunks
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
    """)
    
    results = db.execute(vector_query, {
        "embedding": str(query_embedding), # pgvector uses a string bracket format '[1.0, 0.5, ...]'
        "top_k": top_k
    }).fetchall()
    
    if not results:
        return {"chunks": [], "graph_entities": []}
        
    chunk_ids = [str(row[0]) for row in results]
    chunk_texts = [row[2] for row in results]
    
    # 3. Graph Traversal using Apache AGE
    db.execute(text("LOAD 'age';"))
    db.execute(text("SET search_path = ag_catalog, \"$user\", public;"))
    db.commit()
    
    graph_entities = set()
    
    for c_id in chunk_ids:
        safe_c_id = f"chunk_{c_id}"
        
        # Cypher query to find neighboring entities connected to this chunk
        cypher_q = f"""
        SELECT * FROM cypher('medical_kg', $$
            MATCH (c {{id: '{safe_c_id}'}})-[r]->(e)
            RETURN e.name, e.type, type(r)
        $$) as (name agtype, type agtype, relation agtype);
        """
        try:
            graph_res = db.execute(text(cypher_q)).fetchall()
            for row in graph_res:
                # Agtype returns strings wrapped in quotes
                e_name = str(row[0]).strip('"')
                e_type = str(row[1]).strip('"')
                e_rel = str(row[2]).strip('"')
                graph_entities.add(f"{e_name} ({e_type})")
        except Exception as e:
            print(f"Error querying graph for chunk {c_id}: {e}")
            db.rollback()
            
    return {
        "chunks": chunk_texts,
        "graph_entities": list(graph_entities)
    }

def format_prompt_context(hybrid_results: dict) -> str:
    """
    Formats the raw hybrid context into a structured prompt block for the LLM.
    """
    context_str = "--- RAW TEXT CONTEXT ---\n"
    for i, chunk in enumerate(hybrid_results["chunks"]):
        context_str += f"[Excerpt {i+1}]: {chunk}\n\n"
        
    context_str += "--- KNOWLEDGE GRAPH CONTEXT (Related Entities) ---\n"
    if hybrid_results["graph_entities"]:
        for ent in hybrid_results["graph_entities"]:
            context_str += f"- {ent}\n"
    else:
        context_str += "No additional graph context found.\n"
        
    return context_str
