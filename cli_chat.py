import asyncio
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live

from agent.agents import medical_agent
from agent.db_utils import initialize_database, close_database
from agent.graph_utils import initialize_graph, close_graph
from agent.models import AgentDependencies

console = Console()

async def interactive_chat():
    console.clear()
    console.print(Panel.fit(
        "[bold blue]Medical Edge-Native Agentic RAG[/bold blue]\n"
        "[dim]Private, Local, and Secure Graph + Vector Search[/dim]",
        border_style="blue"
    ))
    
    with console.status("[bold green]Initializing secure connection to local databases...[/bold green]", spinner="dots"):
        await initialize_database()
        await initialize_graph()
        
    console.print("[bold green]v All systems online.\n")
    console.print("Type your medical query below. Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.\n")
    
    session_id = "cli-session-" + os.urandom(4).hex()
    deps = AgentDependencies(session_id=session_id)
    
    try:
        while True:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
                
            if not user_input.strip():
                continue
                
            # Show a thinking spinner while agent processes
            with console.status("[bold cyan]Agent is thinking...[/bold cyan]", spinner="bouncingBar"):
                try:
                    # Run the agent synchronously to avoid the ollama stream bug
                    result = await medical_agent.run(user_input, deps=deps)
                    response_text = result.output
                except Exception as e:
                    response_text = f"**Error:** {str(e)}"
            
            # Print the AI's response formatted as Markdown
            console.print("\n[bold purple]Medical Assistant[/bold purple]")
            console.print(Panel(Markdown(response_text), border_style="purple"))
            console.print()
            
    except KeyboardInterrupt:
        console.print("\n[dim]Received keyboard interrupt.[/dim]")
    finally:
        with console.status("[bold yellow]Safely closing database connections...[/bold yellow]"):
            await close_database()
            await close_graph()
        console.print("[bold green]Goodbye![/bold green]")

if __name__ == "__main__":
    # Windows specific event loop policy to avoid ProactorEventLoop issues on exit
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(interactive_chat())
