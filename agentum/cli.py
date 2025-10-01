# agentum/cli.py
import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .workflow import Workflow

app = typer.Typer(help="Agentum CLI - Run agentic workflows")
console = Console()


@app.command()
def run(
    script_path: str = typer.Argument(
        ..., help="Path to the Python script containing the workflow"
    ),
    initial_state: Optional[str] = typer.Option(
        None, "--state", "-s", help="Initial state as JSON string"
    ),
    thread_id: Optional[str] = typer.Option(
        None, "--thread-id", "-t", help="Thread ID for state persistence"
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Stream workflow execution in real-time"
    ),
):
    """Run an agentum workflow from a Python script."""
    script_file = Path(script_path)

    if not script_file.exists():
        console.print(f"[red]Error: Script '{script_path}' not found.[/red]")
        raise typer.Exit(1)

    if not script_file.suffix == ".py":
        console.print(f"[red]Error: Script must be a Python file (.py).[/red]")
        raise typer.Exit(1)

    try:
        # Import the script
        import importlib.util

        spec = importlib.util.spec_from_file_location("workflow_script", script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for a workflow variable
        workflow = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, Workflow):
                workflow = attr
                break

        if not workflow:
            console.print(
                f"[red]Error: No Workflow instance found in '{script_path}'.[/red]"
            )
            console.print(
                "[yellow]Make sure your script defines a workflow variable.[/yellow]"
            )
            raise typer.Exit(1)

        # Parse initial state
        import json

        if initial_state:
            try:
                state = json.loads(initial_state)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in initial state: {e}[/red]")
                raise typer.Exit(1)
        else:
            state = {}

        # Run the workflow
        if stream:
            asyncio.run(_run_streaming(workflow, state, thread_id))
        else:
            asyncio.run(_run_workflow(workflow, state, thread_id))

    except Exception as e:
        console.print(f"[red]Error running workflow: {e}[/red]")
        raise typer.Exit(1)


async def _run_workflow(workflow: Workflow, state: dict, thread_id: Optional[str]):
    """Run a workflow and print the result."""
    console.print(f"[green]Running workflow '{workflow.name}'...[/green]")

    try:
        result = await workflow.arun(state, thread_id=thread_id)

        console.print("\n[bold green]Workflow completed successfully![/bold green]")
        console.print("\n[bold]Final State:[/bold]")

        import json

        console.print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        console.print(f"[red]Workflow failed: {e}[/red]")
        raise


async def _run_streaming(workflow: Workflow, state: dict, thread_id: Optional[str]):
    """Run a workflow with streaming output."""
    console.print(f"[green]Streaming workflow '{workflow.name}'...[/green]")

    try:
        async for event in workflow.astream(state, thread_id=thread_id):
            console.print(f"[blue]Event: {event}[/blue]")

        console.print("\n[bold green]Workflow stream completed![/bold green]")

    except Exception as e:
        console.print(f"[red]Workflow stream failed: {e}[/red]")
        raise


@app.command()
def version():
    """Show the agentum version."""
    from . import __version__

    console.print(f"Agentum version: [bold green]{__version__}[/bold green]")


@app.command()
def init(
    name: str = typer.Argument(..., help="Name of the workflow"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
):
    """Initialize a new agentum workflow project."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a basic workflow template
    template = f'''# {name}.py
import os
from dotenv import load_dotenv
from agentum import Agent, State, Workflow, tool, GoogleLLM

load_dotenv()

# Define a tool
@tool
def example_tool(query: str) -> str:
    """An example tool that processes queries."""
    return f"Processed: {{query}}"

# Define state
class {name}State(State):
    input: str
    output: str = ""

# Create agent
agent = Agent(
    name="{name}Agent",
    system_prompt="You are a helpful assistant.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[example_tool]
)

# Create workflow
workflow = Workflow(name="{name}", state={name}State)

workflow.add_task(
    name="process",
    agent=agent,
    instructions="Process the input: {{input}}",
    output_mapping={{"output": "output"}}
)

workflow.set_entry_point("process")
workflow.add_edge("process", workflow.END)

# Run the workflow
if __name__ == "__main__":
    result = workflow.run({{"input": "Hello, world!"}})
    print("Result:", result["output"])
'''

    script_path = output_path / f"{name}.py"
    script_path.write_text(template)

    console.print(f"[green]Created workflow template: {script_path}[/green]")
    console.print(f"[blue]Run it with: agentum run {script_path}[/blue]")


if __name__ == "__main__":
    app()
