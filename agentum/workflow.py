# agentum/workflow.py
import asyncio
from typing import Any, Callable, Dict, Optional, Type

from rich.console import Console

# Import moved to _compile method to avoid circular import
from .state import State

console = Console()


class Workflow:
    """
    The main orchestrator for defining, compiling, and running an agentic graph.
    """

    END = "__end__"

    def __init__(
        self, name: str, state: Type[State], persistence: Optional[str] = None
    ):
        self.name = name
        self.state_model = state
        self.persistence = persistence
        self.tasks = {}
        self.edges = []
        self.entry_point = None
        self._compiled_graph = None  # Add a cache for the compiled graph
        self.event_listeners = {}  # NEW: dictionary to hold event listeners
        console.print(f"✨ Workflow '{self.name}' initialized.", style="bold green")

    def on(self, event: str):
        """A decorator to register a listener for a workflow event."""

        def decorator(func: Callable):
            if event not in self.event_listeners:
                self.event_listeners[event] = []
            self.event_listeners[event].append(func)
            return func

        return decorator

    async def _emit(self, event: str, **kwargs):
        """Asynchronously calls all registered listeners for an event."""
        if event in self.event_listeners:
            for listener in self.event_listeners[event]:
                await listener(**kwargs)

    def add_task(
        self,
        name: str,
        agent: Optional[Any] = None,
        tool: Optional[Callable] = None,
        instructions: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        output_mapping: Optional[Dict[str, str]] = None,  # NEW PARAMETER
    ):
        """Adds a node (a unit of work) to the workflow."""
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already exists.")
        self.tasks[name] = {
            "agent": agent,
            "tool": tool,
            "instructions": instructions,
            "inputs": inputs,
            "output_mapping": output_mapping or {name: "output"},  # Default mapping
        }
        console.print(f"  - Task added: [cyan]{name}[/cyan]")

    def add_edge(self, source: str, target: str):
        """Defines a direct connection from one task to another."""
        self.edges.append((source, target))
        console.print(f"  - Edge added: [cyan]{source}[/cyan] -> [cyan]{target}[/cyan]")

    def add_conditional_edges(self, source: str, path: Callable, paths: Dict[str, str]):
        """Defines a branching point based on the output of a path function."""
        console.print(f"  - Conditional Edge added from [cyan]{source}[/cyan]")
        # Store this complex edge information for the compiler
        self.edges.append({"source": source, "path": path, "paths": paths})

    def set_entry_point(self, task_name: str):
        """Sets the starting task for the workflow."""
        self.entry_point = task_name
        console.print(f"  - Entry point set to: [cyan]{task_name}[/cyan]")

    def _compile(self):
        """Compiles the workflow into a runnable graph, caching the result."""
        if not self._compiled_graph:
            console.print(
                "\n🔧 [bold]Compiling workflow into an executable graph...[/bold]",
                style="blue",
            )
            from .engine import GraphCompiler

            compiler = GraphCompiler(self)
            self._compiled_graph = compiler.compile()
            console.print("✅ [bold]Compilation successful.[/bold]", style="green")
        return self._compiled_graph

    def run(self, initial_state: Dict) -> Dict:
        """
        Synchronous wrapper for the async execution of the workflow.
        """
        return asyncio.run(self.arun(initial_state))

    async def arun(self, initial_state: Dict) -> Dict:
        """
        Asynchronously executes the workflow with the given initial state.
        """
        await self._emit("workflow_start", workflow_name=self.name, state=initial_state)

        console.print(
            f"\n🚀 [bold]Running workflow '{self.name}'...[/bold]", style="yellow"
        )

        runnable_graph = self._compile()

        # Use the async 'ainvoke' to run the graph
        final_state = await runnable_graph.ainvoke(initial_state)

        console.print("\n🏁 [bold]Workflow finished.[/bold]", style="yellow")

        await self._emit("workflow_finish", workflow_name=self.name, state=final_state)
        return final_state
