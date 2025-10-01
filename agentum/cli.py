# agentum/cli.py
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .exceptions import TaskConfigurationError, WorkflowDefinitionError
from .workflow import Workflow

app = typer.Typer(help="Agentum CLI - Run agentic workflows")
console = Console()


def make_layout() -> Layout:
    """Define the structure for the live display."""
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="log"),
        Layout(name="final_state", size=10),
    )
    layout["header"].update(
        Panel(
            Text("🚀 Agentum Live Tracer", style="bold green", justify="center"),
            title="[bold magenta]STATUS[/bold magenta]",
            border_style="green",
            height=3,
        )
    )
    layout["final_state"].update(
        Panel(
            Text("Waiting for completion...", style="dim"),
            title="[bold yellow]Final State[/bold yellow]",
            border_style="yellow",
        )
    )
    return layout


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
    """Run a workflow with rich, real-time streaming output."""

    # Initialize the rich layout and log content
    layout = make_layout()
    log_content = Text("", style="dim")

    # Define color scheme for log entries
    TASK_COLOR = "bold cyan"
    AGENT_COLOR = "bold magenta"
    TOOL_COLOR = "bold blue"
    RESULT_COLOR = "green"

    def log(message: str, style: str = "white"):
        """Append a message to the log pane."""
        nonlocal log_content
        log_content.append(Text(f"{message}\n", style=style))
        layout["log"].update(
            Panel(log_content, title="[bold]Workflow Log[/bold]", border_style="white")
        )

    log(f"Starting workflow: {workflow.name}", "bold yellow")
    log(f"Initial State: {list(state.keys())}", "dim")

    try:
        with Live(layout, screen=False, refresh_per_second=4) as live:
            async for event in workflow.astream(state, thread_id=thread_id):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        break

                    # 1. Detect and Log the Completion of the Task
                    log(f"• [bold white]Task: {node_name}[/] finished.", TASK_COLOR)

                    # 2. Extract and Log the State Update
                    if node_output:
                        update_keys = list(node_output.keys())
                        log(
                            f"  ↳ [dim]Updated State Keys:[/dim] {', '.join(update_keys)}",
                            "dim",
                        )

                    # 3. Handle Special Agent/Tool Events (Observability)
                    # We can't easily capture the inner events from astream in this LangGraph structure,
                    # so we will use the global event listener system as a better, more direct way.
                    # For a clean CLI tracer, the focus is on Node I/O.
                    pass

                    # 4. Update the final state display
                    state.update(node_output)

            # After loop breaks, display final status
            final_state = state

            log("🏁 Workflow finished.", "bold green")

            # Update the final state panel
            final_state_syntax = Syntax(
                json.dumps(final_state, indent=2, default=str),
                "json",
                theme="monokai",
                line_numbers=True,
            )
            layout["final_state"].update(
                Panel(
                    final_state_syntax,
                    title="[bold green]Final State[/bold green]",
                    border_style="green",
                )
            )

            # Keep the final display visible for a moment
            await asyncio.sleep(1)

    except Exception as e:
        log(f"❌ Workflow stream failed: {e}", "bold red")
        console.print(f"[red]Error: {e}[/red]")
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


@app.command()
def validate(
    script_path: str = typer.Argument(..., help="Path to the Python script to validate")
):
    """Validate an agentum workflow script without running it."""
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

        # Validate the workflow
        console.print(f"[blue]Validating workflow '{workflow.name}'...[/blue]")

        # Check if workflow has tasks
        if not workflow.tasks:
            console.print("[red]❌ Error: Workflow has no tasks defined.[/red]")
            raise typer.Exit(1)

        # Check if entry point is set
        if not workflow.entry_point:
            console.print("[red]❌ Error: No entry point set for workflow.[/red]")
            raise typer.Exit(1)

        # Check if entry point exists
        if workflow.entry_point not in workflow.tasks:
            console.print(
                f"[red]❌ Error: Entry point '{workflow.entry_point}' not found in tasks.[/red]"
            )
            raise typer.Exit(1)

        # Check for disconnected nodes
        connected_tasks = set()
        for task_name in workflow.tasks.keys():
            connected_tasks.add(task_name)
            # Check edges
            for edge in workflow.edges:
                if edge[0] == task_name:  # edges are tuples (from, to)
                    connected_tasks.add(edge[1])

        disconnected = set(workflow.tasks.keys()) - connected_tasks
        if disconnected:
            console.print(
                f"[yellow]⚠️  Warning: Disconnected tasks: {', '.join(disconnected)}[/yellow]"
            )

        console.print("[green]✅ Workflow validation passed![/green]")
        console.print(f"[blue]Tasks: {len(workflow.tasks)}[/blue]")
        console.print(f"[blue]Edges: {len(workflow.edges)}[/blue]")
        console.print(f"[blue]Entry point: {workflow.entry_point}[/blue]")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def graph(
    script_path: str = typer.Argument(..., help="Path to the Python script"),
    output_file: str = typer.Option(
        "workflow_graph.png", "--output", "-o", help="Output file for the graph"
    ),
):
    """Generate a visual representation of the workflow graph."""
    script_file = Path(script_path)

    if not script_file.exists():
        console.print(f"[red]Error: Script '{script_path}' not found.[/red]")
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
            raise typer.Exit(1)

        # Generate graph visualization
        console.print(
            f"[blue]Generating graph for workflow '{workflow.name}'...[/blue]"
        )

        try:
            import graphviz
        except ImportError:
            console.print(
                "[red]Error: graphviz package not installed. Install with: pip install graphviz[/red]"
            )
            raise typer.Exit(1)

        # Create a new graph
        dot = graphviz.Digraph(comment=workflow.name)
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

        # Add nodes
        for task_name in workflow.tasks.keys():
            if task_name == workflow.entry_point:
                dot.node(task_name, f"🚀 {task_name}", fillcolor="lightgreen")
            else:
                dot.node(task_name, task_name)

        # Add edges
        for edge in workflow.edges:
            if isinstance(edge, tuple):  # Regular edge (from, to)
                if edge[1] == "__end__":
                    dot.node("__end__", "🏁 END", fillcolor="lightcoral")
                    dot.edge(edge[0], "__end__")
                else:
                    dot.edge(edge[0], edge[1])
            elif isinstance(edge, dict):  # Conditional edge
                source = edge["source"]
                for path_name, target in edge["paths"].items():
                    if target == "__end__":
                        dot.node("__end__", "🏁 END", fillcolor="lightcoral")
                        dot.edge(source, "__end__", label=path_name, style="dashed")
                    else:
                        dot.edge(source, target, label=path_name, style="dashed")

        # Render the graph
        output_path = Path(output_file)
        dot.render(output_path.with_suffix(""), format="png", cleanup=True)

        console.print(
            f"[green]✅ Graph saved to: {output_path.with_suffix('.png')}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Graph generation failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
