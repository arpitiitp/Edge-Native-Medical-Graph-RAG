import re
import networkx as nx
from sqlalchemy.orm import Session
from sqlalchemy import text

def sanitize(val: str) -> str:
    """Removes characters that interfere with SQLAlchemy param binding or Cypher syntax."""
    if not val:
        return ""
    val = str(val).replace("'", "")
    val = re.sub(r'[%\{\}\\"\$]', '', val)
    return val

def save_graph_to_db(db: Session, G: nx.MultiDiGraph):
    """
    Takes a NetworkX graph generated from text chunks and saves the 
    nodes and edges into PostgreSQL using the Apache AGE extension.
    """
    # 1. Ensure the Graph namespace exists
    db.execute(text("LOAD 'age';"))
    db.execute(text("SET search_path = ag_catalog, \"$user\", public;"))
    db.commit()
    
    try:
        db.execute(text("SELECT create_graph('medical_kg');"))
        db.commit()
    except Exception as e:
        # Graph might already exist, which is fine
        db.rollback()

    try:
        # 2. Iterate nodes and create AGE Vertices
        for node_id, data in G.nodes(data=True):
            node_type = sanitize(data.get('type', 'Unknown'))
            name = sanitize(data.get('name', ''))
            safe_node_id = sanitize(node_id)

            # Cypher query to merge nodes (avoiding duplicates)
            # We purposefully omit chunk_text to avoid Cypher string interpolation crashes.
            # The full chunk text is already stored in the relational vector table.
            query = f"""
            SELECT * FROM cypher('medical_kg', $$
                MERGE (n:{node_type} {{id: '{safe_node_id}', name: '{name}'}})
                RETURN n
            $$) as (v agtype);
            """
            db.execute(text(query))
            
        # 3. Iterate edges and create AGE Edges
        for u, v, data in G.edges(data=True):
            rel_label = sanitize(data.get('label', 'RELATED_TO'))
            safe_u = sanitize(u)
            safe_v = sanitize(v)
            
            # Cypher query to link existing nodes
            edge_query = f"""
            SELECT * FROM cypher('medical_kg', $$
                MATCH (a {{id: '{safe_u}'}}), (b {{id: '{safe_v}'}})
                MERGE (a)-[r:{rel_label}]->(b)
                RETURN r
            $$) as (e agtype);
            """
            db.execute(text(edge_query))

        db.commit()
        return {"status": "success", "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    except Exception as e:
        db.rollback()
        raise e
