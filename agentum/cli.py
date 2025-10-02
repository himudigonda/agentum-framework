import asyncio
import json
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
    script_file = Path(script_path)
    if not script_file.exists():
        console.print(f"[red]Error: Script '{script_path}' not found.[/red]")
        raise typer.Exit(1)
    if not script_file.suffix == ".py":
        console.print("[red]Error: Script must be a Python file (.py).[/red]")
        raise typer.Exit(1)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("workflow_script", script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
        import json

        if initial_state:
            try:
                state = json.loads(initial_state)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in initial state: {e}[/red]")
                raise typer.Exit(1)
        else:
            state = {}
        if stream:
            asyncio.run(_run_streaming(workflow, state, thread_id))
        else:
            asyncio.run(_run_workflow(workflow, state, thread_id))
    except Exception as e:
        console.print(f"[red]Error running workflow: {e}[/red]")
        raise typer.Exit(1)


async def _run_workflow(workflow: Workflow, state: dict, thread_id: Optional[str]):
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
    layout = make_layout()
    log_content = Text("", style="dim")
    TASK_COLOR = "bold cyan"

    def log(message: str, style: str = "white"):
        nonlocal log_content
        log_content.append(Text(f"{message}\n", style=style))
        layout["log"].update(
            Panel(log_content, title="[bold]Workflow Log[/bold]", border_style="white")
        )

    log(f"Starting workflow: {workflow.name}", "bold yellow")
    log(f"Initial State: {list(state.keys())}", "dim")
    final_state = {}  # Initialize
    try:
        with Live(layout, screen=False, refresh_per_second=4):
            async for event in workflow.astream(state, thread_id=thread_id):
                # Check for the final state event first
                if "__end__" in event:
                    final_state = event["__end__"]
                    break  # Stop processing after the end

                # The rest of the events are for logging node completion
                for node_name, node_output in event.items():
                    log(f"• [bold white]Task: {node_name}[/] finished.", TASK_COLOR)
                    if node_output:
                        update_keys = list(node_output.keys())
                        log(
                            f"  ↳ [dim]Updated State Keys:[/dim] {', '.join(update_keys)}",
                            "dim",
                        )

            # Now, final_state is guaranteed to be the correct one
            log("🏁 Workflow finished.", "bold green")
            if final_state:
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
            await asyncio.sleep(1)
    except Exception as e:
        log(f"❌ Workflow stream failed: {e}", "bold red")
        console.print(f"[red]Error: {e}[/red]")
        raise


@app.command()
def version():
    from . import __version__

    console.print(f"Agentum version: [bold green]{__version__}[/bold green]")


@app.command()
def init(
    name: str = typer.Argument(..., help="Name of the workflow"),
    output_dir: str = typer.Option(".", "--output", "-o", help="Output directory"),
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    template = f'# {name}.py\nimport os\nfrom dotenv import load_dotenv\nfrom agentum import Agent, State, Workflow, tool, GoogleLLM\n\nload_dotenv()\n\n@tool\ndef example_tool(query: str) -> str:\n    """An example tool that processes queries."""\n    return f"Processed: {{query}}"\n\nclass {name}State(State):\n    input: str\n    output: str = ""\n\nagent = Agent(\n    name="{name}Agent",\n    system_prompt="You are a helpful assistant.",\n    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY),\n    tools=[example_tool]\n)\n\nworkflow = Workflow(name="{name}", state={name}State)\n\nworkflow.add_task(\n    name="process",\n    agent=agent,\n    instructions="Process the input: {{input}}",\n    output_mapping={{"output": "output"}}\n)\n\nworkflow.set_entry_point("process")\nworkflow.add_edge("process", workflow.END)\n\nif __name__ == "__main__":\n    result = workflow.run({{"input": "Hello, world!"}})\n    print("Result:", result["output"])\n'
    script_path = output_path / f"{name}.py"
    script_path.write_text(template)
    console.print(f"[green]Created workflow template: {script_path}[/green]")
    console.print(f"[blue]Run it with: agentum run {script_path}[/blue]")


@app.command()
def validate(
    script_path: str = typer.Argument(..., help="Path to the Python script to validate")
):
    script_file = Path(script_path)
    if not script_file.exists():
        console.print(f"[red]Error: Script '{script_path}' not found.[/red]")
        raise typer.Exit(1)
    if not script_file.suffix == ".py":
        console.print("[red]Error: Script must be a Python file (.py).[/red]")
        raise typer.Exit(1)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("workflow_script", script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
        console.print(f"[blue]Validating workflow '{workflow.name}'...[/blue]")
        if not workflow.tasks:
            console.print("[red]❌ Error: Workflow has no tasks defined.[/red]")
            raise typer.Exit(1)
        if not workflow.entry_point:
            console.print("[red]❌ Error: No entry point set for workflow.[/red]")
            raise typer.Exit(1)
        if workflow.entry_point not in workflow.tasks:
            console.print(
                f"[red]❌ Error: Entry point '{workflow.entry_point}' not found in tasks.[/red]"
            )
            raise typer.Exit(1)
        connected_tasks = set()
        for task_name in workflow.tasks.keys():
            connected_tasks.add(task_name)
            for edge in workflow.edges:
                if edge[0] == task_name:
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
    script_file = Path(script_path)
    if not script_file.exists():
        console.print(f"[red]Error: Script '{script_path}' not found.[/red]")
        raise typer.Exit(1)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("workflow_script", script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
        dot = graphviz.Digraph(comment=workflow.name)
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")
        for task_name in workflow.tasks.keys():
            if task_name == workflow.entry_point:
                dot.node(task_name, f"🚀 {task_name}", fillcolor="lightgreen")
            else:
                dot.node(task_name, task_name)
        for edge in workflow.edges:
            if isinstance(edge, tuple):
                if edge[1] == "__end__":
                    dot.node("__end__", "🏁 END", fillcolor="lightcoral")
                    dot.edge(edge[0], "__end__")
                else:
                    dot.edge(edge[0], edge[1])
            elif isinstance(edge, dict):
                source = edge["source"]
                for path_name, target in edge["paths"].items():
                    if target == "__end__":
                        dot.node("__end__", "🏁 END", fillcolor="lightcoral")
                        dot.edge(source, "__end__", label=path_name, style="dashed")
                    else:
                        dot.edge(source, target, label=path_name, style="dashed")
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
