import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from agent.db_utils import db_pool, initialize_database

console = Console()

async def view_vectors():
    console.print(Panel.fit("[bold cyan]PostgreSQL pgvector Database Viewer[/bold cyan]"))
    
    await initialize_database()
    
    async with db_pool.acquire() as conn:
        # Get count
        count = await conn.fetchval("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        console.print(f"[bold green]Total Chunks with Vectors:[/bold green] {count}\n")
        
        # Fetch the first 3 chunks with their embeddings
        query = """
        SELECT c.chunk_index, d.title, c.content, c.embedding::text as vector_str
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        LIMIT 3
        """
        results = await conn.fetch(query)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Document")
        table.add_column("Chunk")
        table.add_column("Text Preview", width=40)
        table.add_column("Vector Array (Truncated)", width=50)
        
        for row in results:
            doc_title = row['title']
            idx = str(row['chunk_index'])
            text = row['content'][:80] + "..." if len(row['content']) > 80 else row['content']
            
            # The vector comes back as a string like "[0.0123, -0.0456, ...]"
            # We'll just show the first 5 numbers so it doesn't flood the terminal
            vec_str = row['vector_str']
            # Parse it roughly to show the structure
            elements = vec_str.strip('[]').split(',')
            truncated_vec = "[" + ", ".join(elements[:5]) + ", ... (768 total dims)]"
            
            table.add_row(doc_title, idx, text, truncated_vec)
            
        console.print(table)
        
    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(view_vectors())
