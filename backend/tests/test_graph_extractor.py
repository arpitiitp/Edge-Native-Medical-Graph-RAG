from backend.services.graph_extractor import extract_entities_and_relationships, build_chunk_graph
import networkx as nx

def test_extraction():
    test_text = "Lisinopril causes dry cough"
    print(f"Testing Text: '{test_text}'")
    
    entities, rels = extract_entities_and_relationships(test_text)
    print("\n--- Extracted Entities ---")
    for e in entities:
        print(f"- {e['name']} ({e['type']})")
        
    print("\n--- Inferred Relationships ---")
    for source, target, rel in rels:
        print(f"- {source} -[{rel}]-> {target}")
        
    G = build_chunk_graph(999, test_text)
    print(f"\n--- NetworkX Graph Summary ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    test_extraction()
