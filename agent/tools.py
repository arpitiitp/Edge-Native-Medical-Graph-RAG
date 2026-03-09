"""
Search tools for the Pydantic AI Agents.
"""

import logging
from typing import List, Dict, Any, Optional

from agent.db_utils import hybrid_search, vector_search
from agent.graph_utils import graph_client
from ingestion.embedder import create_embedder

logger = logging.getLogger(__name__)

# Initialize a global embedder for tools
_embedder = create_embedder()


async def search_medical_guidelines(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search vector database for medical documents, guidelines, and raw patient report text.
    
    Args:
        query: The search query (e.g., "What are the guidelines for hypertension?")
        limit: Maximum number of chunks to return
        
    Returns:
        List of matching document chunks with metadata and similarity scores
    """
    logger.info(f"Executing semantic vector search for: '{query}'")
    try:
        # Embed the query
        query_embedding = await _embedder.embed_query(query)
        
        # Perform hybrid search (we can adjust text_weight if needed)
        results = await hybrid_search(
            embedding=query_embedding,
            query_text=query,
            limit=limit,
            text_weight=0.3
        )
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "source": r.get('document_source', 'Unknown'),
                "title": r.get('document_title', 'Unknown'),
                "content": r.get('content', ''),
                "score": round(r.get('combined_score', 0), 4)
            })
            
        return formatted_results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return [{"error": str(e)}]


async def search_knowledge_graph(entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search the Neo4j Knowledge Graph for specific medical entities (Conditions, Medications, Personnel, Facilities).
    Use this to find relationships or see contexts where specific entities are mentioned.
    
    Args:
        entity_name: Node name to search for (e.g., "Hypertension", "Aspirin", "Dr. Smith")
        limit: Maximum number of contexts to return
        
    Returns:
        List of graph context paths
    """
    logger.info(f"Executing knowledge graph search for entity: '{entity_name}'")
    
    # Query looks for the entity node and the chunks that mention it
    cypher_query = """
    MATCH (c:Chunk)-[:MENTIONS]->(e)
    WHERE toLower(e.name) CONTAINS toLower($entity)
    RETURN 
        e.name AS entity_name, 
        labels(e)[0] AS entity_type, 
        c.content AS context, 
        c.source AS source
    LIMIT $limit
    """
    
    try:
        results = await graph_client.execute_query(
            cypher_query, 
            {"entity": entity_name, "limit": limit}
        )
        return results
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return [{"error": str(e)}]

