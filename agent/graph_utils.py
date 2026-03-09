"""
Graph utilities for native Neo4j integration.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from neo4j import AsyncGraphDatabase

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DirectNeo4jClient:
    """Manages simple Neo4j knowledge graph operations directly."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """Initialize Neo4j client."""
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
            
        self.driver = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Neo4j driver."""
        if self._initialized:
            return
            
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            await self.driver.verify_connectivity()
            self._initialized = True
            logger.info("DirectNeo4jClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize direct Neo4j client: {e}")
            raise
    
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            self._initialized = False
            logger.info("Neo4j direct client closed")
            
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts."""
        if not self._initialized:
            await self.initialize()
            
        async with self.driver.session() as session:
            try:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
            except Exception as e:
                logger.error(f"Cypher query execution failed: {e}\nQuery: {query}\nParams: {parameters}")
                raise
    
    async def clear_graph(self):
        """Clear all nodes and relationships."""
        await self.execute_query("MATCH (n) DETACH DELETE n")
        logger.warning("Knowledge graph cleared (Native Neo4j)")

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get counts of nodes and relationships."""
        try:
            nodes = await self.execute_query("MATCH (n) RETURN count(n) as count")
            edges = await self.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
            return {
                "nodes": nodes[0]["count"],
                "relationships": edges[0]["count"]
            }
        except Exception as e:
            return {"error": str(e)}

# Global Neo4j direct client instance
graph_client = DirectNeo4jClient()

async def initialize_graph():
    await graph_client.initialize()

async def close_graph():
    await graph_client.close()

async def test_graph_connection() -> bool:
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False