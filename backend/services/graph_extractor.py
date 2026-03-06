import spacy
from spacy.pipeline import EntityRuler
import networkx as nx
from typing import List, Tuple, Dict

print("Loading Spacy NER Model...")
try:
    nlp = spacy.load("en_core_web_sm")
    # Add custom rules to extract Medical terms from the generic Blood Report for the prototype
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "blood"}, {"LOWER": "cancer"}]},
        {"label": "MEDICAL_TEST", "pattern": [{"LOWER": "complete"}, {"LOWER": "blood"}, {"LOWER": "count"}]},
        {"label": "MEDICAL_TEST", "pattern": [{"LOWER": "haemoglobin"}]},
        {"label": "MEDICAL_TEST", "pattern": [{"LOWER": "wbc"}, {"LOWER": "count"}]},
        {"label": "MEDICAL_TEST", "pattern": [{"LOWER": "platelet"}, {"LOWER": "count"}]},
    ]
    ruler.add_patterns(patterns)
    print("Spacy loaded successfully with custom Medical Ruler.")
except Exception as e:
    print(f"Failed to load Spacy model: {e}")
    nlp = None

def extract_entities_and_relationships(text: str) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, str]]]:
    """
    Given an input string, uses standard Spacy to extract entities and infers simple relationships.
    Returns: A list of entity dictionaries and a list of relationships (source_id, target_id, relation).
    """
    if nlp is None:
        return [], []

    doc = nlp(text)
    entities = []
    
    # Store standard identifiers for deduplication based on exact text
    for ent in doc.ents:
        # ent.label_ will typically be 'DISEASE' or 'CHEMICAL' with the bc5cdr model
        entity = {
            "id": ent.text.lower().strip(),
            "name": ent.text.strip(),
            "type": ent.label_
        }
        entities.append(entity)

    # Basic relationship inference (In a real scenario, this involves a biomedical RE model 
    # like specific transformer RE models. For this BXP, we do a proximity/co-occurrence linkage)
    relationships = []
    
    # We create a CAUSES or TREATS edge based on the order and types (Chemical -> Disease = TREATS/CAUSES)
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i]
            e2 = entities[j]
            
            # Simple heuristic mapping for the prototype
            if e1['type'] == 'CHEMICAL' and e2['type'] == 'DISEASE':
                relationships.append((e1['id'], e2['id'], "TREATS_OR_CAUSES"))
            elif e1['type'] == 'DISEASE' and e2['type'] == 'CHEMICAL':
                relationships.append((e2['id'], e1['id'], "TREATS_OR_CAUSES"))

    return entities, relationships

def build_chunk_graph(chunk_id: int, text: str) -> nx.MultiDiGraph:
    """
    Transforms text into a directed graph structure.
    Adds a 'Chunk' origin node that metadata links to all extracted entities.
    """
    G = nx.MultiDiGraph()
    chunk_node_id = f"chunk_{chunk_id}"
    
    # Add the source text chunk as a node itself
    G.add_node(chunk_node_id, type="CHUNK", text=text)

    entities, relationships = extract_entities_and_relationships(text)

    for ent in entities:
        # Add entity node and connect it back to the chunk
        G.add_node(ent["id"], type=ent["type"], name=ent["name"])
        G.add_edge(chunk_node_id, ent["id"], label="MENTIONS")

    for e1_id, e2_id, rel in relationships:
        G.add_edge(e1_id, e2_id, label=rel)

    return G
