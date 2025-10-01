# agentum/workflow.py
from typing import Any, Dict, Type

from rich.console import Console

# Import moved to _compile method to avoid circular import
from .state import State

console = Console()


class Workflow:
    """
    The main orchestrator for defining, compiling, and running an agentic graph.
    """

    END = "__end__"

    def __init__(self, name: str, state: Type[State], persistence: str = None):
        self.name = name
        self.state_model = state
        self.persistence = persistence
        self.tasks = {}
        self.edges = []
        self.entry_point = None
        self._compiled_graph = None  # Add a cache for the compiled graph
        console.print(f"✨ Workflow '{self.name}' initialized.", style="bold green")

    def add_task(
        self,
        name: str,
        agent: Any = None,
        tool: callable = None,
        instructions: str = None,
        inputs: Dict = None,
    ):
        """Adds a node (a unit of work) to the workflow."""
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already exists.")
        self.tasks[name] = {
            "agent": agent,
            "tool": tool,
            "instructions": instructions,
            "inputs": inputs,
        }
        console.print(f"  - Task added: [cyan]{name}[/cyan]")

    def add_edge(self, source: str, target: str):
        """Defines a direct connection from one task to another."""
        self.edges.append((source, target))
        console.print(f"  - Edge added: [cyan]{source}[/cyan] -> [cyan]{target}[/cyan]")

    def add_conditional_edges(self, source: str, path: callable, paths: Dict[str, str]):
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
            # Import here to avoid circular import
            from .engine import GraphCompiler
            
            console.print(
                "\n🔧 [bold]Compiling workflow into an executable graph...[/bold]",
                style="blue",
            )
            compiler = GraphCompiler(self)
            self._compiled_graph = compiler.compile()
            console.print("✅ [bold]Compilation successful.[/bold]", style="green")
        return self._compiled_graph

    def run(self, initial_state: Dict) -> Dict:
        """
        Executes the workflow with the given initial state.
        """
        console.print(
            f"\n🚀 [bold]Running workflow '{self.name}'...[/bold]", style="yellow"
        )

        # Get the compiled, runnable graph
        runnable_graph = self._compile()

        # LangGraph's invoke method runs the graph
        final_state = runnable_graph.invoke(initial_state)

        console.print("\n🏁 [bold]Workflow finished.[/bold]", style="yellow")
        return final_state
