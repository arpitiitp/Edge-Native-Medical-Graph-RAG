"""
Knowledge graph builder for extracting entities and relationships.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import asyncio
import re
import aiohttp

from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import DirectNeo4jClient
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import DirectNeo4jClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph_client = DirectNeo4jClient()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    
    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 10 
    ) -> Dict[str, Any]:
        """
        Add document chunks and entities to the knowledge graph manually.
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"nodes_created": 0, "errors": []}
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")
        
        nodes_created = 0
        errors = []
        
        cypher_query = """
        MERGE (d:Document {source: $doc_source})
        SET d.title = $doc_title
        
        MERGE (c:Chunk {source: $doc_source, index: $chunk_index})
        SET c.content = $chunk_content
        MERGE (c)-[:PART_OF]->(d)
        
        FOREACH (cond IN $conditions | 
            MERGE (e:Condition {name: cond})
            MERGE (c)-[:MENTIONS]->(e)
        )
        FOREACH (med IN $medications | 
            MERGE (m:Medication {name: med})
            MERGE (c)-[:MENTIONS]->(m)
        )
        FOREACH (pers IN $personnel | 
            MERGE (p:Personnel {name: pers})
            MERGE (c)-[:MENTIONS]->(p)
        )
        FOREACH (fac IN $facilities | 
            MERGE (f:Facility {name: fac})
            MERGE (c)-[:MENTIONS]->(f)
        )
        """
        
        for i, chunk in enumerate(chunks):
            try:
                entities = chunk.metadata.get("entities", {})
                params = {
                    "doc_title": document_title,
                    "doc_source": document_source,
                    "chunk_index": chunk.index,
                    "chunk_content": chunk.content,
                    "conditions": entities.get("conditions", []),
                    "medications": entities.get("medications", []),
                    "personnel": entities.get("personnel", []),
                    "facilities": entities.get("facilities", [])
                }
                
                await self.graph_client.execute_query(cypher_query, params)
                nodes_created += 1
                logger.info(f"✓ Mapped chunk {chunk.index} to graph ({nodes_created}/{len(chunks)})")
                
            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "chunks_mapped": nodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Graph building complete: {nodes_created} chunks mapped, {len(errors)} errors")
        return result
    

    
    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_conditions: bool = True,
        extract_medications: bool = True,
        extract_personnel: bool = True
    ) -> List[DocumentChunk]:
        """
        Extract entities from chunks using a local LLM and JSONL prompting.
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks using Ollama JSONL")
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("LLM_CHOICE", "llama3.2")
        
        enriched_chunks = []
        
        async with aiohttp.ClientSession() as session:
            for i, chunk in enumerate(chunks):
                entities = {
                    "conditions": [],
                    "medications": [],
                    "personnel": [],
                    "facilities": []
                }
                
                content = chunk.content
                
                # Construct JSONL extraction prompt
                system_prompt = """You are a strict medical data extraction assistant.
Extract the following entities from the text:
- CONDITIONS (diseases, symptoms, diagnoses)
- MEDICATIONS (drugs, treatments, prescriptions)
- PERSONNEL (doctors, nurses, medical staff)
- FACILITIES (hospitals, clinics, wards, departments)

Output ONLY line-delimited JSON (JSONL). Each line MUST be a single JSON object.
Do NOT wrap the output in markdown code blocks.
Do NOT output any conversational text.

Example output:
{"entity_type": "PERSONNEL", "name": "Dr. Smith"}
{"entity_type": "CONDITION", "name": "Hypertension"}
{"entity_type": "MEDICATION", "name": "Lisinopril"}
{"entity_type": "FACILITY", "name": "General Hospital"}"""

                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"TEXT TO EXTRACT FROM:\n{content}"}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0  # Deterministic output
                    }
                }
                
                try:
                    async with session.post(f"{ollama_base_url}/api/chat", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            output_text = result.get("message", {}).get("content", "")
                            
                            # Parse JSONL output
                            for line in output_text.strip().split('\n'):
                                line = line.strip()
                                # Clean up common SLM mistakes (like markdown wrapping)
                                if line.startswith("```"): continue
                                if not line: continue
                                
                                try:
                                    data = json.loads(line)
                                    e_type = data.get("entity_type", "").upper()
                                    name = data.get("name", "").strip()
                                    
                                    if not name: continue
                                    
                                    if e_type == "CONDITION" and extract_conditions:
                                        entities["conditions"].append(name.lower())
                                    elif e_type == "MEDICATION" and extract_medications:
                                        entities["medications"].append(name.lower())
                                    elif e_type == "PERSONNEL" and extract_personnel:
                                        entities["personnel"].append(name)
                                    elif e_type == "FACILITY":
                                        entities["facilities"].append(name)
                                        
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSONL line: {line}")
                        else:
                            logger.error(f"Ollama API returned status {response.status}")
                except Exception as e:
                    logger.error(f"Error calling Ollama API for chunk {i}: {e}")
                
                # Deduplicate
                entities["conditions"] = list(set(entities["conditions"]))
                entities["medications"] = list(set(entities["medications"]))
                entities["personnel"] = list(set(entities["personnel"]))
                entities["facilities"] = list(set(entities["facilities"]))
                
                # Create enriched chunk
                enriched_chunk = DocumentChunk(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "entities": entities,
                        "entity_extraction_date": datetime.now().isoformat(),
                        "extraction_method": "ollama_jsonl"
                    },
                    token_count=chunk.token_count
                )
                
                # Preserve embedding if it exists
                if hasattr(chunk, 'embedding'):
                    enriched_chunk.embedding = chunk.embedding
                
                enriched_chunks.append(enriched_chunk)
                logger.info(f"Processed chunk {i+1}/{len(chunks)} with Ollama")
        
        logger.info("Entity extraction complete")
        return enriched_chunks
    
    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


class SimpleEntityExtractor:
    """Simple rule-based entity extractor as fallback."""
    
    def __init__(self):
        """Initialize extractor."""
        self.condition_patterns = [
            r'\b(?:hypertension|diabetes|cancer|tumor|infection|asthma|arthritis|pneumonia|covid-19|virus|bacterial)\b',
            r'\b(?:pain|fever|cough|nausea|fatigue|inflammation|fracture|syndrome|disorder)\b'
        ]
        
        self.medication_patterns = [
            r'\b(?:aspirin|ibuprofen|acetaminophen|antibiotic|penicillin|vaccine|insulin|statin|antidepressant)\b',
            r'\w+(?:cillin|mycin|statin|pril|olol|pine)\b'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using patterns."""
        entities = {
            "conditions": [],
            "medications": []
        }
        
        # Extract conditions
        for pattern in self.condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["conditions"].extend(matches)
        
        # Extract medications
        for pattern in self.medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["medications"].extend(matches)
        
        # Remove duplicates and clean up
        entities["conditions"] = list(set([m.lower() for m in entities["conditions"]]))
        entities["medications"] = list(set([m.lower() for m in entities["medications"]]))
        
        return entities


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()


# Example usage
async def main():
    """Example usage of the graph builder."""
    from .chunker import ChunkingConfig, create_chunker
    
    # Create chunker and graph builder
    config = ChunkingConfig(chunk_size=300, use_semantic_splitting=False)
    chunker = create_chunker(config)
    graph_builder = create_graph_builder()
    
    sample_text = """
    Patient John Doe presented to the emergency department with severe chest pain and
    shortness of breath. He has a history of hypertension and type 2 diabetes.
    The attending physician, Dr. Smith, ordered a full cardiac panel and an EKG.
    
    The patient was administered 324mg of Aspirin and sublingual Nitroglycerin.
    He was later admitted to the Cardiology wing at Springfield General Hospital
    for further observation.
    """
    
    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text,
        title="Patient Medical Report",
        source="example.md"
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Extract entities
    enriched_chunks = await graph_builder.extract_entities_from_chunks(chunks)
    
    for i, chunk in enumerate(enriched_chunks):
        print(f"Chunk {i}: {chunk.metadata.get('entities', {})}")
    
    # Add to knowledge graph
    try:
        result = await graph_builder.add_document_to_graph(
            chunks=enriched_chunks,
            document_title="Patient Medical Report",
            document_source="example.md",
            document_metadata={"topic": "Medical", "date": "2024"}
        )
        
        print(f"Graph building result: {result}")
        
    except Exception as e:
        print(f"Graph building failed: {e}")
    
    finally:
        await graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())