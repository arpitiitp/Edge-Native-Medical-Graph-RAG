import asyncio
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from agent.tools import search_medical_guidelines, search_knowledge_graph
from agent.graph_utils import initialize_graph, close_graph

console = Console()

async def run_demonstration():
    console.print(Panel.fit("[bold cyan]Medical RAG Architecture Demonstration[/bold cyan]\n[white]Vector Semantic Search vs Neo4j Knowledge Graph[/white]"))
    
    await initialize_graph()
    
    # --- SCENARIO 1: The Vector Weakness ---
    console.print("\n[bold yellow]Scenario 1: Complex Relational Query[/bold yellow]")
    console.print("[italic]Question: 'What medications is John Doe taking, and what conditions are they for?'[/italic]")
    
    # 1A: Try with Vector Search
    console.print("\n[bold red]Attempt 1: Standard Vector Search (Semantics)[/bold red]")
    vector_results = await search_medical_guidelines("John Doe medications and conditions", limit=3)
    
    table_vec = Table(show_header=True, header_style="bold red")
    table_vec.add_column("Score")
    table_vec.add_column("Document Text Snippet")
    
    for r in vector_results:
        snippet = r['content'][:150] + "..." if len(r['content']) > 150 else r['content']
        table_vec.add_row(str(r['score']), snippet)
    
    console.print(table_vec)
    console.print("[dim white]Note how Vector Search just returns paragraphs. The LLM has to read all these paragraphs and try to piece together the answer itself, which often leads to hallucinations or missed facts.[/dim white]")
    
    # 1B: Try with Graph Search
    console.print("\n[bold green]Attempt 2: Knowledge Graph Search (Relational Traversals)[/bold green]")
    graph_results = await search_knowledge_graph("John Doe", limit=5)
    
    table_graph = Table(show_header=True, header_style="bold green")
    table_graph.add_column("Exact Entity Found")
    table_graph.add_column("Entity Type")
    table_graph.add_column("Graph Context Hook")
    
    for r in graph_results:
        entity = r.get('entity_name', 'Unknown')
        etype = r.get('entity_type', 'Unknown')
        context = r.get('context', '')[:80] + "..."
        table_graph.add_row(entity, etype, context)
        
    console.print(table_graph)
    console.print("[dim white]Notice how the Knowledge Graph bypassed reading paragraphs and instantly traversed the edges to find EXACT medical entities (Persons, Conditions, Medications) explicitly linked to John Doe.[/dim white]")
    
    await close_graph()

if __name__ == "__main__":
    asyncio.run(run_demonstration())
